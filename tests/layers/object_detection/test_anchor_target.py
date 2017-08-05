import keras.backend
import numpy

import keras_rcnn.layers


class TestAnchorTarget:
    def test_call(self):
        gt_boxes = keras.backend.variable(numpy.random.random((91, 4)))
        im_info = keras.backend.variable([[224, 224, 3]])
        scores = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 2)))

        proposal_target = keras_rcnn.layers.AnchorTarget()

        proposal_target.call([scores, gt_boxes, im_info])
