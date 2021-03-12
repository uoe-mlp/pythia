from typing import List, Union, Optional, Callable, Any
import tensorflow as tf
import numpy as np


class OutputObserver(tf.keras.callbacks.Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, model, X_train, Y_hat, epochs):
        self.model = model.seq_model
        self.X_train = X_train
        self.Y_hat: np.ndarray = Y_hat if isinstance(Y_hat, np.ndarray) else Y_hat.numpy()
        self.batch_num = 0
        self.epochs = epochs
        self.active = False

    def on_batch_end(self, epoch, logs={}):
        if self.active:
            self.Y_hat[self.batch_num : self.batch_num + 1, :, :] = self.model.predict(self.X_train[self.batch_num : self.batch_num + 1, :, :])
            self.batch_num += 1

    def on_epoch_start(self, epoch, logs={}):
        if epoch == self.epochs - 1:
            self.batch_num = 0
            self.active = True
        else:
            self.active = False

class LSTMChalvatzisTF(object):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]]):
        super(LSTMChalvatzisTF, self).__init__()
        
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout for x in hidden_size_list]

        self.input_size: int = input_size
        self.hidden_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size

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
                    dropout=d,
                    recurrent_dropout=0,
                    input_shape=(window_size, hidden_size_list[i-1])))

        self.seq_model.add(tf.keras.layers.Flatten(
                input_shape=(window_size, hidden_size_list[-1])))
        self.seq_model.add(tf.keras.layers.Dense(output_size * window_size, 
                input_shape=(input_size * hidden_size_list[-1],)))
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