import keras.backend
import keras_rcnn.layers
import numpy


def test_call_classification():
    anchors = 9
    layer = keras_rcnn.layers.ClassificationLoss(anchors=anchors)

    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 2 * anchors)))
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, anchors)))

    numpy.testing.assert_array_equal(layer.call([y_true, y_pred]), y_pred)

    assert len(layer.losses) == 1

    expected_loss = -numpy.log(0.5)
    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)


def test_call_regression():
    anchors = 9
    layer = keras_rcnn.layers.RegressionLoss(anchors=anchors)
    y_true = keras.backend.variable(numpy.random.random((91, 4)))
    scores = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 2)))
    deltas = keras.backend.variable(numpy.random.random((1, 14, 14, 9 * 4)))

    expected_loss = numpy.power(0.5, 3)

    labels, bbox_reg_targets = keras_rcnn.layers.AnchorTarget()([scores, y_true, image])
    #numpy.testing.assert_array_equal(layer.call([y_true, deltas, scores]), y_pred)
    assert numpy.isclose(keras.backend.eval(layer.call([bbox_reg_targets, deltas, labels])), expected_loss)

    assert len(layer.losses) == 1

    assert numpy.isclose(keras.backend.eval(layer.losses[0]), expected_loss)
