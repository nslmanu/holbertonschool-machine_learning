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
