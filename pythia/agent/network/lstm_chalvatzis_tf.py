from typing import List, Union, Optional, Callable, Any
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow import keras


class OutputObserver(tf.keras.callbacks.Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, model, X_train, Y_hat, epochs, batch_size):
        if type(model) is LSTMChalvatzisTFFunc:
            self.model = model
        else:
            self.model = model.seq_model
        self.X_train = X_train
        self.Y_hat: np.ndarray = Y_hat if isinstance(Y_hat, np.ndarray) else Y_hat.numpy()
        self.batch_num = 0
        self.epochs = epochs
        self.active = False
        self.batch_size: int = batch_size

    def on_batch_end(self, epoch, logs={}):
        if self.active:
            i_from = self.batch_num * self.batch_size
            i_to = min((self.batch_num + 1) * self.batch_size, self.Y_hat.shape[0])
            self.Y_hat[i_from : i_to, :, :] = self.model.predict(self.X_train[i_from : i_to, :, :])
            self.batch_num += 1

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == self.epochs - 1:
            self.batch_num = 0
            self.active = True
        else:
            self.active = False

class MaskedDense(tf.keras.layers.Layer):
    def __init__(self, input_shape, output_shape, kernel_regularizer=None,  name=None):
        super(MaskedDense, self).__init__(name=name)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.w = self.add_weight("kernel",
                                  shape=[input_shape, output_shape],
                                  regularizer=self.kernel_regularizer)
        lengths = [int(i / (input_shape / output_shape))  for i in range(input_shape)]
        self.mask = 1 - (tf.sequence_mask(lengths, output_shape, dtype=tf.dtypes.float32))

    def __call__(self, x):
        y = tf.matmul(x, tf.math.multiply(self.mask, self.w))
        return y

class LSTMChalvatzisTFFunc(keras.Model):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]],  masked: bool, l2: float=0.0):
        super(LSTMChalvatzisTFFunc, self).__init__()
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout for x in hidden_size_list]

        self.input_size: int = input_size
        self.hidden_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size
        self.l2: float = l2
        self.masked: bool = masked


        self.input_layer = tf.keras.layers.InputLayer(input_shape=(window_size, input_size))
        self.lstm_layers: List[tf.keras.layers.LSTM] = []

        for i, (hs, d) in enumerate(zip(hidden_size_list, dropout_list)):
            if i == 0:
                self.lstm_layers.append(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=d,
                    recurrent_dropout=0,
                    input_shape=(window_size,input_size)))
            else:
                self.lstm_layers.append(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=0,
                    recurrent_dropout=0,
                    input_shape=(window_size, hidden_size_list[i-1])))

        self.flatten_layer = tf.keras.layers.Flatten(input_shape=(window_size, hidden_size_list[-1]))

        reg = tf.keras.regularizers.L2(self.l2)

        self.dense_layer = None

        if not self.masked:
            self.dense_layer = tf.keras.layers.Dense(output_size * window_size, use_bias=False,
                    input_shape=(window_size * hidden_size_list[-1],), kernel_regularizer=reg)
        else:
            self.dense_layer = MaskedDense(input_shape=(window_size * hidden_size_list[-1]),
                    output_shape = output_size * window_size, kernel_regularizer=reg)
        self.reshape_layer = tf.keras.layers.Reshape((window_size, output_size),
                                input_shape=(output_size * window_size, 1,))


    def call(self, inputs):
        x = self.input_layer(inputs)

        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return self.reshape_layer(x)






class LSTMChalvatzisTF(object):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]],  masked: bool, l2: float=0.0):
        super(LSTMChalvatzisTF, self).__init__()
        
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout for x in hidden_size_list]

        self.input_size: int = input_size
        self.hidden_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size
        self.l2: float = l2
        self.masked: bool = masked

        self.seq_model = tf.keras.Sequential()
        self.seq_model.add(
            tf.keras.layers.InputLayer(input_shape=(window_size, input_size))
        )
        for i, (hs, d) in enumerate(zip(hidden_size_list, dropout_list)):
            if i == 0:
                self.seq_model.add(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=d,
                    recurrent_dropout=0,
                    input_shape=(window_size,input_size)))
            else:
                self.seq_model.add(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=0,
                    recurrent_dropout=0,
                    input_shape=(window_size, hidden_size_list[i-1])))

        self.seq_model.add(tf.keras.layers.Flatten(
                input_shape=(window_size, hidden_size_list[-1])))

        reg = tf.keras.regularizers.L2(self.l2)

        if not self.masked:
            self.seq_model.add(tf.keras.layers.Dense(output_size * window_size, use_bias=False,
                    input_shape=(window_size * hidden_size_list[-1],), kernel_regularizer=reg))
        else:
            self.seq_model.add(MaskedDense(input_shape=(window_size * hidden_size_list[-1]),
                    output_shape = output_size * window_size, kernel_regularizer=reg))
        self.seq_model.add(
            tf.keras.layers.Reshape((window_size, output_size),
                input_shape=(output_size * window_size, 1,))
        )
        
    def call(self, inputs):
        return self.seq_model.call(inputs)
    
    def compile(self, *args, **kwargs):
        return self.seq_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.seq_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.seq_model.predict(*args, **kwargs)

    def build(self, *args, **kwargs):
        return self.seq_model.build(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.seq_model.summary(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.seq_model.evaluate(*args, **kwargs)
    
    def detach_model(self) -> Any:
        m = self.seq_model
        self.seq_model = None
        return m
    
    def copy_model(self) -> Any:
        return tf.keras.models.clone_model(self.seq_model, input_tensors=self.seq_model.input)

    def attach_model(self, model) -> None:
        self.seq_model = model

    def set_weights(self, *args, **kwargs):
        return self.seq_model.set_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.seq_model.get_weights(*args, **kwargs)