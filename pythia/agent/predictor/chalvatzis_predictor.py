from __future__ import annotations
from typing import Dict, Tuple, Dict, List, Any, Optional
from abc import ABC, abstractclassmethod
import numpy as np
import tensorflow as tf

from pythia.utils import ArgsParser
from pythia.agent.network import LSTMChalvatzisTF

from .predictor import Predictor


class ChalvatzisPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, window_size: int, hidden_size: int, dropout: float, all_hidden: bool,
                 epochs: int, batch_size: int, shuffle: bool, predict_returns: bool, 
                 initial_learning_rate: float, learning_rate_decay: float, loss: str='mse'):
        super(ChalvatzisPredictor, self).__init__(input_size, output_size, predict_returns)
        
        self.window_size: int = window_size
        self.hidden_size: int = hidden_size
        self.dropout: float = dropout
        self.all_hidden: bool = all_hidden
        self.epochs: int = epochs
        self.batch_size: int = batch_size if batch_size is not None else 1
        self.shuffle: bool = shuffle
        self.model = LSTMChalvatzisTF(
            input_size=input_size,  window_size=window_size, hidden_size=[hidden_size, hidden_size], output_size=output_size,
            dropout=[dropout, dropout])
        self.lr_schedule: tf.keras.optimizers.Schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1,
            decay_rate=learning_rate_decay,
            staircase=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.loss: str = loss
        self.model.compile(self.optimizer, self.loss, ['mae'])


    @property
    def last_hidden(self) -> bool:
        return not self.all_hidden

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        epochs: int = ArgsParser.get_or_default(params, 'epochs', 100)
        shuffle: bool = ArgsParser.get_or_default(params, 'shuffle', False)
        batch_size: int = ArgsParser.get_or_default(params, 'batch_size', 1)
        hidden_size: int = ArgsParser.get_or_default(params, 'hidden_size', 64)
        dropout: float = ArgsParser.get_or_default(params, 'dropout', 0)
        learning_rate_decay: float = ArgsParser.get_or_default(params, 'learning_rate_decay', 1)
        initial_learning_rate: float = ArgsParser.get_or_default(params, 'initial_learning_rate', 0.001)
        # all_hidden: bool = ArgsParser.get_or_default(params, 'all_hidden', True)
        all_hidden: bool = True
        predict_returns: bool = ArgsParser.get_or_default(params, 'predict_returns', False)
        window_size: int = ArgsParser.get_or_default(params, 'window_size', 5)

        return ChalvatzisPredictor(input_size=input_size, output_size=output_size, window_size=window_size, hidden_size=hidden_size, 
            epochs=epochs, predict_returns=predict_returns, shuffle=shuffle, batch_size=batch_size, dropout=dropout, all_hidden=all_hidden, learning_rate_decay=learning_rate_decay,
            initial_learning_rate=initial_learning_rate)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, **kwargs):
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y np.ndarray by an index (a day).
        """
        splits = [X.shape[0]]
        if X_val is not None and Y_val is not None:
            splits.append(X.shape[0] + X_val.shape[0])
            X = np.concatenate([X, X_val], axis=0)
            Y = np.concatenate([Y, Y_val], axis=0)
        
        if self.predict_returns:
            X = X[:-1,:]
            Y = Y[1:,:] / Y[:-1,:] - 1
        else:
            X = X[:-1,:]
            Y = Y[1:,:]

        data = self.__create_sequences(X, Y, splits)
        X_train, Y_train = data[0]
        if X_val is not None and Y_val is not None:
            X_val, Y_val = data[1]
        else:
            X_val, Y_val = X_train, Y_train

        self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val))

    def __create_sequences(self, X: np.ndarray, Y: np.ndarray, splits: List[int]=[]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Args:
            X (np.ndarray): [description]
            Y (np.ndarray): [description]
            splits (List[int], optional): if the output X, Y should  be splitted, this contains the index of the first X value of any new split. Defaults to [].
        """
        X_list: List[List[np.ndarray]] = [[]]
        Y_list: List[List[np.ndarray]] = [[]]

        for i in range(X.shape[0] - self.window_size):
            if (i + self.window_size - 1) < splits[0]:
                X_list[-1].append(X[i:i + self.window_size,:])
                Y_list[-1].append(Y[i:i + self.window_size,:])
            else:
                X_list.append([])
                Y_list.append([])
                splits = splits[1:]
                X_list[-1].append(X[i:i + self.window_size,:])
                Y_list[-1].append(Y[i:i + self.window_size,:])

        data: List[Tuple[np.ndarray, np.ndarray]] = []
        
        for x_ls, y_ls in zip(X_list, Y_list):
            if self.shuffle:
                reindex = list(range(len(x_ls)))
                np.random.shuffle(reindex)
                x_ls = [i for _,i in sorted(zip(reindex,x_ls))]
                y_ls = [i for _,i in sorted(zip(reindex,y_ls))]
                
            data.append((np.array(x_ls), np.array(y_ls)))

        return data

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: prediction and conviction
        """
        x = X[-self.window_size:, :]
        output = self.model.predict(np.array([x]))
        return output, np.abs(output)
