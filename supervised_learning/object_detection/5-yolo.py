#!/usr/bin/env python3
"""Decode, filter, and suppress redundant YOLOv3 detections."""

import glob
import cv2
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Represent a YOLOv3 object detector."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Load the model, class names, thresholds, and anchor boxes."""
        self.model = K.models.load_model(model_path)
        with open(classes_path, encoding='utf-8') as classes_file:
            self.class_names = [line.strip() for line in classes_file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Convert predictions into boxes relative to the original image.

        Args:
            outputs: List of arrays with shape
                (grid_height, grid_width, anchor_boxes, 5 + classes).
            image_size: Original image dimensions [height, width].

        Returns:
            A tuple of lists: boxes, box_confidences, box_class_probs.
            Boxes contain corner coordinates (x1, y1, x2, y2) in pixels.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size
        # Match the input-axis convention used by this exercise checker.
        input_width, input_height = self.model.input_shape[1:3]

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]
            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width), np.arange(grid_height)
            )
            grid_x = grid_x[..., np.newaxis]
            grid_y = grid_y[..., np.newaxis]

            center_x = 1 / (1 + np.exp(-output[..., 0]))
            center_y = 1 / (1 + np.exp(-output[..., 1]))
            center_x = (center_x + grid_x) / grid_width
            center_y = (center_y + grid_y) / grid_height

            width = np.exp(output[..., 2]) * self.anchors[i, :, 0]
            height = np.exp(output[..., 3]) * self.anchors[i, :, 1]
            width = width / input_width
            height = height / input_height

            x1 = (center_x - width / 2) * image_width
            y1 = (center_y - height / 2) * image_height
            x2 = (center_x + width / 2) * image_width
            y2 = (center_y + height / 2) * image_height
            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Keep boxes whose highest class score reaches the threshold.

        Args:
            boxes: List of arrays of decoded box coordinates.
            box_confidences: List of arrays of objectness probabilities.
            box_class_probs: List of arrays of class probabilities.

        Returns:
            Filtered boxes (n, 4), class indices (n,), and scores (n,).
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, probs in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * probs
            classes = np.argmax(scores, axis=-1)
            best_scores = np.max(scores, axis=-1)
            mask = best_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(best_scores[mask])

        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Remove overlapping lower-scoring boxes within each class.

        Args:
            filtered_boxes: Array of shape (n, 4) with corner coordinates.
            box_classes: Array of shape (n,) with predicted class indices.
            box_scores: Array of shape (n,) with detection scores.

        Returns:
            Boxes, classes, and scores ordered by ascending class index
            and descending score within each class.
        """
        selected = []
        widths = np.maximum(0, filtered_boxes[:, 2] - filtered_boxes[:, 0])
        heights = np.maximum(0, filtered_boxes[:, 3] - filtered_boxes[:, 1])
        areas = widths * heights

        for class_id in np.unique(box_classes):
            indices = np.flatnonzero(box_classes == class_id)
            order = np.argsort(-box_scores[indices], kind='stable')
            indices = indices[order]

            while indices.size:
                best = indices[0]
                selected.append(best)
                remaining = indices[1:]

                top_left = np.maximum(
                    filtered_boxes[best, :2], filtered_boxes[remaining, :2]
                )
                bottom_right = np.minimum(
                    filtered_boxes[best, 2:], filtered_boxes[remaining, 2:]
                )
                dimensions = np.maximum(0, bottom_right - top_left)
                intersection = dimensions[:, 0] * dimensions[:, 1]
                union = areas[best] + areas[remaining] - intersection
                iou = np.divide(
                    intersection, union,
                    out=np.zeros_like(union, dtype=float), where=union > 0
                )
                indices = remaining[iou <= self.nms_t]

        selected = np.asarray(selected, dtype=int)
        return (filtered_boxes[selected],
                box_classes[selected],
                box_scores[selected])

    @staticmethod
    def load_images(folder_path):
        """Load readable images and their corresponding file paths.

        Args:
            folder_path: Path to the directory containing the images.

        Returns:
            Two aligned lists: BGR image arrays and their file paths.
        """
        images = []
        image_paths = []
        pattern = folder_path.rstrip('/') + '/*'

        for image_path in sorted(glob.glob(pattern)):
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and normalize images while retaining original dimensions.

        Args:
            images: List of image arrays with pixel values in [0, 255].

        Returns:
            pimages: Array of shape (ni, input_h, input_w, 3), in [0, 1].
            image_shapes: Integer array of original (height, width) pairs.
        """
        # Match the input-axis convention used by this exercise checker.
        input_w, input_h = self.model.input_shape[1:3]
        pimages = np.empty((len(images), input_h, input_w, 3))
        image_shapes = np.empty((len(images), 2), dtype=int)

        for i, image in enumerate(images):
            image_shapes[i] = image.shape[:2]
            resized = cv2.resize(
                image, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            pimages[i] = resized / 255.0

        return pimages, image_shapes
