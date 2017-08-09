import keras.backend
import keras.engine

import keras_rcnn.backend


class ProposalTarget(keras.layers.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.

    # Arguments

    # Input shape

    # Output shape
    """

    def __init__(self, fg_fraction=0.5, batchsize=256, num_images=1, num_classes=1, **kwargs):
        self.fg_fraction = fg_fraction
        self.batchsize = batchsize
        self.num_images = num_images
        self.num_classes = num_classes
        super(ProposalTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProposalTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # GT boxes (x1, y1, x2, y2)
        # and other times after box coordinates -- normalize to one format
        proposals, bounding_boxes, labels = inputs

        proposals = keras.backend.concatenate((proposals, bounding_boxes), 1)

        rois_per_image = self.batchsize / self.num_images
        fg_rois_per_image = keras.backend.round(self.fg_fraction * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets

        labels, rois, bbox_targets = keras_rcnn.backend.sample_rois(proposals, bounding_boxes, labels, fg_rois_per_image, rois_per_image, self.num_classes)

        return [rois, labels, bbox_targets]

    def compute_output_shape(self, input_shape):
        return [(None,), (None, 1), (None, 4)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None]
