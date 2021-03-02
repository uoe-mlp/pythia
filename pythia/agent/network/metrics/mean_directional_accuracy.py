import tensorflow as tf


class MeanDirectionalAccuracy(tf.keras.metrics.Metric):

  def __init__(self, name='mean_directional_accuracy', **kwargs):
    super(MeanDirectionalAccuracy, self).__init__(name=name, **kwargs)
    self.mda = self.add_weight(name='mda', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.equal(y_true, y_pred)
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, values.shape)
      values = tf.multiply(values, sample_weight)
    self.mda.assign(tf.reduce_mean(values))

  def result(self):
    return self.mda