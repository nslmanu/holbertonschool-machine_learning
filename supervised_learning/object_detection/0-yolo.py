#!/usr/bin/env python3
"""Initialize a YOLOv3 object detector."""

from tensorflow import keras as K


class Yolo:
    """Represent a YOLOv3 object detector."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the model and detection settings.

        Args:
            model_path: Path to the saved Darknet Keras model.
            classes_path: Path to the file containing class names.
            class_t: Box score threshold for initial filtering.
            nms_t: IoU threshold for non-max suppression.
            anchors: Array of anchor widths and heights with shape
                (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, encoding='utf-8') as classes_file:
            self.class_names = [
                line.strip() for line in classes_file
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
