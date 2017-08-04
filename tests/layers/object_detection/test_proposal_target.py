import keras.backend
import numpy

import keras_rcnn.layers


class TestProposalTarget:
    def test_call(self):
        gt_boxes = keras.backend.variable(numpy.random.random((84, 9 * 4)))
        im_info = (14, 14, 1)

        proposal_target = keras_rcnn.layers.AnchorTarget(allowed_border=0, clobber_positives=False, negative_overlap=0.3, positive_overlap=0.7)

        proposal_target.call([gt_boxes, im_info])
