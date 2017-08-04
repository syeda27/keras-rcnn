import keras.backend
import keras.engine

import keras_rcnn.backend


class AnchorTarget(keras.layers.Layer):
    """Calculate proposal anchor targets and corresponding labels (label: 1 is positive, 0 is negative, -1 is do not care) for ground truth boxes

    # Arguments
        allowed_border: allow boxes to be outside the image by allowed_border pixels
        clobber_positives: if an anchor statisfied by positive and negative conditions given to negative label
        negative_overlap: IoU threshold below which labels should be given negative label
        positive_overlap: IoU threshold above which labels should be given positive label

    # Input shape
        (# of samples, 4), (width of feature map, height of feature map, scale)

    # Output shape
        (# of samples, ), (# of samples, 4)
    """

    def __init__(self, allowed_border=0, clobber_positives=False, negative_overlap=0.3, positive_overlap=0.7, stride=16, **kwargs):
        self.allowed_border = allowed_border
        self.clobber_positives = clobber_positives
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.stride = stride

        super(AnchorTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AnchorTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        scores, gt_boxes, metadata = inputs

        # TODO: Fix usage of batch index
        shape = metadata[0, :2]

        rr, cc = keras.backend.int_shape(scores)[1:-1]

        # 1. Generate proposals from bbox deltas and shifted anchors
        anchors = keras_rcnn.backend.shift((rr, cc), self.stride)

        # only keep anchors inside the image
        indices, anchors = keras_rcnn.backend.inside_image(anchors, metadata[0], self.allowed_border)

        # 2. obtain indices of gt boxes with the greatest overlap, balanced labels
        argmax_overlaps_indices, labels = keras_rcnn.backend.label(anchors, gt_boxes, indices, self.negative_overlap, self.positive_overlap, self.clobber_positives)

        gt_boxes = keras.backend.gather(gt_boxes, argmax_overlaps_indices)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        bbox_reg_targets = keras_rcnn.backend.bbox_transform(anchors, gt_boxes)

        # TODO: Why is bbox_reg_targets' shape (5, ?, 4)? Why is gt_boxes' shape (None, None, 4) and not (None, 4)?
        bbox_reg_targets = keras.backend.reshape(bbox_reg_targets, (-1, 4))

        # TODO: implement inside and outside weights
        return [labels, bbox_reg_targets]

    def compute_output_shape(self, input_shape):
        return [(None,), (None,)]

    def compute_mask(self, inputs, mask=None):
        # unfortunately this is required
        return 2 * [None]
