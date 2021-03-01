from typing import List, Union, Optional, Callable
import tensorflow as tf


# class OutputObserver(tf.keras.callbacks.Callback):
#     """"
#     callback to observe the output of the network
#     """

#     def on_train_begin(self, logs={}):
#         self.epoch = []
#         self.out_log = []

#     def on_epoch_end(self, epoch, logs={}):
#         self.epoch.append(epoch) 
#         self.out_log.append(self.model.layers[-1].output)

class LSTMChalvatzisTF(object):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]]):
        super(LSTMChalvatzisTF, self).__init__()
        
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout]

        self.input_size: int = input_size
        self.hidden_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size

        self.__model = tf.keras.Sequential()
        self.__model.add(
            tf.keras.layers.InputLayer(input_shape=(window_size, input_size))
        )
        for i, (hs, d) in enumerate(zip(hidden_size_list, dropout_list)):
            if i == 0:
                self.__model.add(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=d,
                    recurrent_dropout=0,
                    input_shape=(window_size,input_size)))
            else:
                self.__model.add(tf.keras.layers.LSTM(
                    units=hs, 
                    activation='relu',
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    dropout=d,
                    recurrent_dropout=0,
                    input_shape=(window_size, hidden_size_list[i-1])))

        self.__model.add(tf.keras.layers.Flatten(
                input_shape=(window_size, hidden_size_list[-1])))
        self.__model.add(tf.keras.layers.Dense(output_size * window_size, 
                input_shape=(input_size * hidden_size_list[-1],)))
        self.__model.add(
            tf.keras.layers.Reshape((window_size, output_size),
                input_shape=(output_size * window_size, 1,))
        )
        
    def call(self, inputs):
        return self.__model.call(inputs)
    
    def compile(self, *args, **kwargs):
        return self.__model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.__model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.__model.predict(*args, **kwargs)

    def build(self, *args, **kwargs):
        return self.__model.build(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.__model.summary(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.__model.evaluate(*args, **kwargs)