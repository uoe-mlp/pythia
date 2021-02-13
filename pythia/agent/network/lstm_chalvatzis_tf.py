from typing import List, Tuple, Union, Optional
import tensorflow as tf 
import numpy as np


class LSTMChalvatzisTF(tf.keras.Model):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]]):
        super(LSTMChalvatzisTF, self).__init__()
        
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout]

        self.input_size: int = input_size
        self.hidden_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size

        self.input_block: tf.keras.layers.InputLayer = tf.keras.layers.InputLayer(input_shape=(window_size,input_size))
        self.lstm_blocks: List[tf.keras.Layer] = []
        for hs, d in zip(hidden_size_list, dropout_list):
            self.lstm_blocks.append(tf.keras.layers.LSTM(
                units=hs, 
                activation='relu',
                use_bias=True,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                dropout=d,
                recurrent_dropout=0,
                input_shape=(window_size,input_size)))

        self.flatten_block: tf.keras.layers.Flatten = tf.keras.layers.Flatten()
        self.dense_block: tf.keras.layers.Dense = tf.keras.layers.Dense(output_size * window_size)
        self.reshape_block: tf.keras.layers.Reshape = tf.keras.layers.Reshape((window_size, output_size))
        
    def call(self, inputs):
        x = self.input_block(inputs)
        for lstm_block in self.lstm_blocks:
            x = lstm_block(x)
        x = self.flatten_block(x)
        x = self.dense_block(x)
        x = self.reshape_block(x)
        return x