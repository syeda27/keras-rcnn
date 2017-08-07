import keras.backend
import keras.layers
import keras.models
import numpy

import keras_rcnn.layers


class TestAnchorTarget:
    def test_call(self):
        gt_boxes = keras.backend.variable(numpy.random.random((1, 10000, 4)))
        im_info = keras.backend.variable([[224, 224, 1]])
        scores = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 2)))

        proposal_target = keras_rcnn.layers.AnchorTarget()

        proposal_target.call([scores, gt_boxes, im_info])

    def test_use(self):
        image = keras.layers.Input((224, 224, 3))

        bounding_boxes = keras.layers.Input((None, 4), name="bounding_boxes")

        metadata = keras.layers.Input((3,), name="metadata")

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        y = keras.layers.Conv2D(64, **options)(image)
        y = keras.layers.Conv2D(64, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(128, **options)(y)
        y = keras.layers.Conv2D(128, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)

        y = keras.layers.Conv2D(512, **options)(y)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1))(y)
        scores = keras.layers.Conv2D(9 * 2, (1, 1), activation="sigmoid")(y)

        labels, bbox_reg_targets = keras_rcnn.layers.AnchorTarget()([scores, bounding_boxes, image])

        classification = keras_rcnn.layers.ClassificationLoss(9)([labels, scores])

        regression = keras_rcnn.layers.RegressionLoss(9)([bbox_reg_targets, deltas, labels])

        model = keras.models.Model([image, bounding_boxes], [classification, regression])

        model.compile("adam", [None, None])

        y_true_image = numpy.random.random((1, 224, 224, 3))
        y_true_bounding_boxes = numpy.random.random((1, 10, 4))
        y_true_metadata = numpy.asarray([[14, 14, 1]])

        model.fit([y_true_image, y_true_bounding_boxes], y=None)
