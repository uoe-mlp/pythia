from __future__ import annotations
from typing import Dict, Tuple, Dict, List, Any, Optional, Callable
from abc import ABC, abstractclassmethod
import numpy as np
from numpy.core.arrayprint import dtype_is_implied
import tensorflow as tf
import copy
import os

from pythia.utils import ArgsParser
from pythia.agent.network import LSTMChalvatzisTF, OutputObserver

from .predictor import Predictor


class ChalvatzisPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, window_size: int, hidden_size: int, dropout: float, all_hidden: bool,
                 epochs: int, iter_per_item: int, shuffle: bool, predict_returns: bool, first_col_cash: bool,
                 initial_learning_rate: float, learning_rate_decay: float, batch_size: int, update_iter_per_item: int, 
                 loss: str='mse', normalize: bool=False, normalize_min: Optional[float]=None, normalize_max: Optional[float]=None):
        super(ChalvatzisPredictor, self).__init__(input_size, output_size, predict_returns, first_col_cash)
        
        self.window_size: int = window_size
        self.hidden_size: int = hidden_size
        self.dropout: float = dropout
        self.all_hidden: bool = all_hidden
        self.epochs: int = epochs
        self.batch_size = batch_size
        self.iter_per_item: int = iter_per_item if iter_per_item is not None else 1
        self.shuffle: bool = shuffle
        self.update_iter_per_item: int = update_iter_per_item
        self.normalize: bool = normalize
        if self.normalize:
            self.normalize_min: float = normalize_min if normalize_min is not None else -1
            self.normalize_max: float = normalize_max if normalize_max is not None else 1
        self.model = LSTMChalvatzisTF(
            input_size=input_size, window_size=window_size, hidden_size=hidden_size, output_size=self.output_size,
            dropout=dropout)
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
        iter_per_item: int = ArgsParser.get_or_default(params, 'iter_per_item', 1) # Number of backpropagations on a single item before jumping to the next one
        hidden_size: int = ArgsParser.get_or_default(params, 'hidden_size', 64)
        dropout: float = ArgsParser.get_or_default(params, 'dropout', 0)
        learning_rate_decay: float = ArgsParser.get_or_default(params, 'learning_rate_decay', 1)
        initial_learning_rate: float = ArgsParser.get_or_default(params, 'initial_learning_rate', 0.001)
        batch_size: int = ArgsParser.get_or_default(params, 'batch_size', 1)
        update_iter_per_item: int = ArgsParser.get_or_default(params, 'update_iter_per_item', iter_per_item)
        normalize_dict: Dict[str, Any] = ArgsParser.get_or_default(params, 'normalize', {})
        if normalize_dict:
            normalize: bool = ArgsParser.get_or_default(normalize_dict, 'active', False)
            if normalize:
                normalize_min: Optional[float] = ArgsParser.get_or_default(normalize_dict, 'min', -1)
                normalize_max: Optional[float] = ArgsParser.get_or_default(normalize_dict, 'max', 1)
            else:
                normalize_min = None
                normalize_max = None
        else:
            normalize = False
            normalize_min = None
            normalize_max = None

        # all_hidden: bool = ArgsParser.get_or_default(params, 'all_hidden', True)
        all_hidden: bool = True
        predict_returns: bool = ArgsParser.get_or_default(params, 'predict_returns', False)
        first_col_cash: bool = ArgsParser.get_or_default(params, 'first_col_cash', False)
        window_size: int = ArgsParser.get_or_default(params, 'window_size', 5)

        return ChalvatzisPredictor(input_size=input_size, output_size=output_size, window_size=window_size, hidden_size=hidden_size, 
            epochs=epochs, predict_returns=predict_returns, first_col_cash=first_col_cash, shuffle=shuffle, iter_per_item=iter_per_item, dropout=dropout, all_hidden=all_hidden, learning_rate_decay=learning_rate_decay,
            batch_size=batch_size, initial_learning_rate=initial_learning_rate, normalize=normalize, normalize_min=normalize_min, normalize_max=normalize_max,
            update_iter_per_item=update_iter_per_item)

    def _inner_fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, epochs_between_validation: Optional[int]=None, val_infra: Optional[List]=None, **kwargs):
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y np.ndarray by an index (a day).
        """
        X_in = X.copy()
        Y_in = Y.copy()
        X_val_in = X_val.copy() if X_val is not None else X_in
        Y_val_in = Y_val.copy() if Y_val is not None else Y_in

        splits = [X.shape[0]]
        if X_val is not None and Y_val is not None:
            splits.append(X.shape[0] + X_val.shape[0])
            X = np.concatenate([X, X_val], axis=0)
            Y = np.concatenate([Y, Y_val], axis=0)
        
        X = X[:-1,:]
        Y = self.prepare_prices(Y)
        
        if self.normalize:
            self.__normalize_fit(X, Y)
            X = self.__normalize_apply_features(X)
            Y = self.__normalize_apply_targets(Y)

        data = self.__create_sequences(X, Y, splits)
        X_train, Y_train = data[0]
        if X_val is not None and Y_val is not None:
            X_val, Y_val = data[1]
        else:
            X_val, Y_val = X_train, Y_train
        
        X_train = np.array([X_train,] * self.iter_per_item).transpose([1,0,2,3]).reshape([X_train.shape[0] * self.iter_per_item] + list(X_train.shape[1:]))
        Y_train = np.array([Y_train,] * self.iter_per_item).transpose([1,0,2,3]).reshape([Y_train.shape[0] * self.iter_per_item] + list(Y_train.shape[1:]))
        X_train = tf.convert_to_tensor(X_train, dtype=tf.dtypes.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.dtypes.float32)
        
        obs = OutputObserver(self.model, X_train, Y_train, self.epochs)
        if (epochs_between_validation is not None):
            loops = np.ceil(self.epochs / float(epochs_between_validation))
            for loop in range(int(loops)):
                if loop == loops - 1:
                    epochs: int = int(self.epochs - loops * epochs_between_validation)
                else:
                    epochs = int(epochs_between_validation)
                self.model.fit(X_train, Y_train, epochs=epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[obs])

                # Predict
                Y_hat = obs.Y_hat[::self.iter_per_item, -1, :]
                if self.normalize:
                    Y_hat = self.__normalize_apply_targets(Y_hat, revert=True)
                if (loop == loops - 1) and (epochs == epochs_between_validation):
                    pass
                else:
                    self.validate(loop, val_infra, Y_hat, X_in, Y_in, X_val_in, Y_val_in)
        else:
            self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[obs])
            # Predict
            Y_hat = obs.Y_hat[::self.iter_per_item, -1, :]
            if self.normalize:
                Y_hat = self.__normalize_apply_targets(Y_hat, revert=True)
        return Y_hat

    def __normalize_fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._normalize_fitted_min_feature: np.ndarray = X.min(axis=0)
        self._normalize_fitted_max_feature: np.ndarray = X.max(axis=0)
        
        # Handling the case where min == max, where the denominator in the normalisation would be 0
        eq = self._normalize_fitted_min_feature == self._normalize_fitted_max_feature
        self._normalize_fitted_min_feature[eq] -=0.5
        self._normalize_fitted_max_feature[eq] +=0.5

        self._normalize_fitted_min_target: np.ndarray = Y.min(axis=0)
        self._normalize_fitted_max_target: np.ndarray = Y.max(axis=0)

        # Handling the case where min == max, where the denominator in the normalisation would be 0
        eq = self._normalize_fitted_min_target == self._normalize_fitted_max_target
        self._normalize_fitted_min_target[eq] -=0.5
        self._normalize_fitted_max_target[eq] +=0.5

    def __normalize_apply_features(self, X: np.ndarray) -> np.ndarray:
        return (X - self._normalize_fitted_min_feature) / (self._normalize_fitted_max_feature - self._normalize_fitted_min_feature) * (self.normalize_max - self.normalize_min) + self.normalize_min

    def __normalize_apply_targets(self, Y: np.ndarray, revert: bool=False) -> np.ndarray:
        if revert:
            return (Y - self.normalize_min) / (self.normalize_max - self.normalize_min) * (self._normalize_fitted_max_target - self._normalize_fitted_min_target) + self._normalize_fitted_min_target
        else:
            return (Y - self._normalize_fitted_min_target) / (self._normalize_fitted_max_target - self._normalize_fitted_min_target) * (self.normalize_max - self.normalize_min) + self.normalize_min

    def __create_sequences(self, X: np.ndarray, Y: np.ndarray, splits: List[int]=[]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Args:
            X (np.ndarray): [description]
            Y (np.ndarray): [description]
            splits (List[int], optional): if the output X, Y should  be splitted, this contains the index of the first X value of any new split. Defaults to [].
        """
        X_list: List[List[np.ndarray]] = [[]]
        Y_list: List[List[np.ndarray]] = [[]]

        for i in range(X.shape[0] - self.window_size + 1):
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

    def _inner_predict(self, X: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: prediction and conviction
        """
        if all_history:
            if self.normalize:
                X = self.__normalize_apply_features(X)
            data = self.__create_sequences(X, X, [X.shape[0]])
            X, _ = data[0]

            output = self.model.predict(X)[:,-1,:]
            if self.normalize:
                output = self.__normalize_apply_targets(output, revert=True)
            return output, np.abs(output)
        else:
            x = X[-self.window_size:, :]
            if self.normalize:
                x = self.__normalize_apply_features(x)
            output = self.model.predict(np.array([x]))[0,-1,:]
            if self.normalize:
                output = self.__normalize_apply_targets(output, revert=True)
            return output, np.abs(output)

    def _inner_update(self, X: np.ndarray, Y: np.ndarray) -> None:
        x = X[-self.window_size:, :]
        y = Y[-self.window_size:, :]
        if self.normalize:
            x = self.__normalize_apply_features(x)
            y = self.__normalize_apply_targets(y)
        data = self.__create_sequences(x, y, [self.window_size])
        x, y = data[0]
        
        X_train = np.array([x,] * self.update_iter_per_item).transpose([1,0,2,3]).reshape([x.shape[0] * self.update_iter_per_item] + list(x.shape[1:]))
        Y_train = np.array([y,] * self.update_iter_per_item).transpose([1,0,2,3]).reshape([y.shape[0] * self.update_iter_per_item] + list(y.shape[1:]))

        self.model.fit(X_train, Y_train, batch_size=1)

    def detach_model(self) -> Any:
        m = self.model.detach_model()
        return m
    
    def copy_model(self) -> Any:
        model = LSTMChalvatzisTF(
            input_size=self.input_size, window_size=self.window_size, hidden_size=self.hidden_size, output_size=self.output_size,
            dropout=self.dropout)
        lr_schedule: tf.keras.optimizers.Schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr_schedule.initial_learning_rate,
            decay_steps=1,
            decay_rate=self.lr_schedule.decay_rate,
            staircase=False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss = self.loss
        model.compile(optimizer, loss, ['mae'])
        model.set_weights(self.model.get_weights()) 
        return model

    def attach_model(self, model) -> None:
        self.model.attach_model(model)

    def validate(self, num, val_infra, prediction, X_train, Y_train, X_val, Y_val) -> None:
        agent = val_infra[0]
        market_execute = val_infra[1]
        timestamps = val_infra[2]
        instruments = val_infra[3]
        journal = val_infra[4]
        train_num = val_infra[5]
        val_num = val_infra[6]
        trader = val_infra[7]

        model_copy = self.copy_model()
        model = self.detach_model()
        predictor: Predictor = copy.deepcopy(self)
        self.attach_model(model)
        predictor.attach_model(model_copy)
        trader.fit(prediction=prediction, conviction=prediction, Y=Y_train, predict_returns=predictor.predict_returns)

        agent.predictor = predictor
        agent.trader = trader
        X = np.concatenate([X_train, X_val], axis=0)
        Y = np.concatenate([Y_train, Y_val], axis=0)

        for i in range(val_num):
            idx = train_num + i
            timestamp = timestamps[idx]
            trade_orders = agent.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
            journal.store_order(trade_orders)
            trade_fills = market_execute(trade_orders, timestamp)
            journal.store_fill(trade_fills)
            agent.update(trade_fills, X[:idx + 1, :], Y[:idx + 2, :])

        journal.run_analytics('train_%d' % num, timestamps[train_num:train_num + val_num], Y_val, instruments)