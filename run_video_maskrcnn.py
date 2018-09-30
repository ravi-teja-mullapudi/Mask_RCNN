from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys, random, time, math
import tensorflow as tf

import cv2
import coco
import utils
import model as modellib

tf.app.flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint file.')
tf.app.flags.DEFINE_string('video_in_path', None, 'Path to input video file.')
tf.app.flags.DEFINE_string('video_out_path', None, 'Path to output video file.')
tf.app.flags.DEFINE_string('detections_path', None, 'Path to output detections file.')
tf.app.flags.DEFINE_integer('max_frames', 100000, 'Maximum number of frames to log.')
tf.app.flags.DEFINE_integer('stride', 5, 'Interval at which detections are computed.')

FLAGS = tf.app.flags.FLAGS

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def draw_boxes(frame, boxes):
    for bidx, bbox in enumerate(boxes):
        iy = int(bbox[0] * frame.shape[0])
        ix = int(bbox[1] * frame.shape[1])
        h = int((bbox[2] - bbox[0]) * frame.shape[0])
        w = int((bbox[3] - bbox[1]) * frame.shape[1])

        cv2.rectangle(frame, (ix, iy),
                                  (ix + w, iy + h),
                                  (0, 255, 0), 8)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def process_detections(detections, mrcnn_mask, image_shape, window):
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = float(image_shape[0]) / (window[2] - window[0])
    w_scale = float(image_shape[1]) / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    return boxes, class_ids, scores, masks

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [num_instances, height, width]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    """
    # Number of instances
    N = boxes.shape[0]

    if N == 0:
        return image.copy()

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.copy()
    for i in range(N):
        color = (0, 0, 255)

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 4)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    return masked_image

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5

def run_model(checkpoint_path, video_in_path, video_out_path, detections_path,
              max_frames, stride):

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir='./logs', config=config)
    model.load_weights(checkpoint_path, by_name=True)

    cap = cv2.VideoCapture(video_in_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = int(cap.get(cv2.CAP_PROP_FPS))

    vid_out = None
    if video_out_path:
        vid_out = cv2.VideoWriter(video_out_path,
                                  cv2.VideoWriter_fourcc('M','J','P','G'),
                                  rate, (width, height))
    count = 0

    frame_detections = {}

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()

        if not ret or count > max_frames:
            break

        if count % stride != 0:
            count = count + 1
            continue

        molded_images, image_metas, windows = model.mold_inputs([frame])

        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rois, rpn_class, rpn_bbox = \
            model.keras_model.predict([molded_images, image_metas], verbose=0)

        zero_ix = np.where(detections[0][:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections[0].shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes, class_ids, scores, masks = process_detections(detections[0],
                                                             mrcnn_mask[0],
                                                             frame.shape,
                                                             windows[0])

        print(boxes.shape, class_ids.shape, scores.shape, masks.shape)

        boxes = boxes.astype(np.float32)
        boxes[:, 0] = boxes[:, 0]/frame.shape[0]
        boxes[:, 2] = boxes[:, 2]/frame.shape[0]
        boxes[:, 1] = boxes[:, 1]/frame.shape[1]
        boxes[:, 3] = boxes[:, 3]/frame.shape[1]

        frame_detections[count] = [boxes, class_ids, scores, masks]
        print(class_ids)

        if vid_out:
            final_rois, final_class_ids, final_scores, final_masks = \
                model.unmold_detections(detections[0], mrcnn_mask[0],
                                        frame.shape, windows[0])

            mask_img = display_instances(frame, final_rois, final_masks,
                                         final_class_ids, class_names,
                                         final_scores)
            vid_out.write(mask_img)

        end = time.time()
        print('time', count, end - start)
        count = count + 1

    if vid_out:
        vid_out.release()

    if detections_path:
        np.save(detections_path, frame_detections)

run_model(FLAGS.checkpoint_path,
          FLAGS.video_in_path,
          FLAGS.video_out_path,
          FLAGS.detections_path,
          FLAGS.max_frames,
          FLAGS.stride)
