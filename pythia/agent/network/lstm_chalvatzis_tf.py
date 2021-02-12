from typing import List, Tuple, Union, Optional
import tensorflow as tf 
import numpy as np


class LSTMChalvatzisTF(object):

    def __init__(self, input_size: int, window_size: int, hidden_size: Union[int, List[int]], output_size: int, dropout: Union[float, List[float]]):
        hidden_size_list: List[int] = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        dropout_list: List[float] = dropout if isinstance(dropout, list) else [dropout]

        self.input_size: List[int] = hidden_size_list
        self.dropout: List[float] = dropout_list
        self.window_size: int = window_size
        self.output_size: int = output_size

        self.model = tf.keras.Sequential()
        for hs, d in zip(hidden_size_list, dropout_list):
            self.model.add(tf.keras.layers.LSTM(
                units=hs, 
                activation='relu',
                use_bias=True,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                dropout=d,
                recurrent_dropout=d,
                input_shape=(window_size,input_size)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(output_size * window_size))
        self.model.add(tf.keras.layers.Reshape((window_size, output_size)))

        opt = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    def fit(self, X: np.array, Y: np.array, X_val: np.array, Y_val: np.array, epochs: int, batch_size: int) -> None:
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

    def evaluate(self, X: np.array, Y: np.array) -> Tuple[float, float]:
        mse, mae = self.model.evaluate(X, Y, verbose=1)
        return (mse, mae)
    
    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def describe(self) -> None:
        print(self.model.summary())