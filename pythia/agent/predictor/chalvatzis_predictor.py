from __future__ import annotations
from typing import Dict, Tuple, Dict, List, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
import copy

from pythia.utils import ArgsParser
from pythia.agent.network import LSTMChalvatzisTF, OutputObserver

from .predictor import Predictor


def custom_loss(y_actual,y_pred,factor):
    mean_square_loss = tf.reduce_mean(tf.losses.mean_squared_error(y_actual, y_pred))
    bce = tf.losses.BinaryCrossentropy()
    cross_entropy = factor *  tf.reduce_mean(bce(
        tf.cast(tf.greater_equal(y_actual, 0.0), tf.float32),
        tf.sigmoid(y_pred)))
        
    loss = tf.add(mean_square_loss, cross_entropy)
    return loss
    
class ChalvatzisPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, window_size: int, hidden_size: int, dropout: float, all_hidden: bool,
                 epochs: int, iter_per_item: int, shuffle: bool, predict_returns: bool, first_column_cash: bool,
                 initial_learning_rate: float, learning_rate_decay: float, batch_size: int, update_iter_per_item: int, masked: bool,
                 loss: str='mse', normalize: bool=False, normalize_min: Optional[float]=None, normalize_max: Optional[float]=None, l2: float=0.0,
                 update_rolling_window: int=1, consume_returns: bool = False):
        super(ChalvatzisPredictor, self).__init__(input_size, output_size, predict_returns, first_column_cash)
        
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
        self.masked: bool = masked
        self.update_rolling_window: int = update_rolling_window
        self.consume_returns: bool = consume_returns
        self.l2: float = l2
        self.loss_name: str = loss
        self.learning_rate_decay: float = learning_rate_decay
        self.initial_learning_rate: float = initial_learning_rate

        if self.normalize:
            self.normalize_min: float = normalize_min if normalize_min is not None else -1
            self.normalize_max: float = normalize_max if normalize_max is not None else 1

        self._initialise_model()
 
    def _initialise_model(self):
        self.model = LSTMChalvatzisTF(
            input_size=self.input_size, window_size=self.window_size, hidden_size=self.hidden_size, output_size=self.output_size,
            dropout=self.dropout, l2=self.l2, masked=self.masked)
        self.lr_schedule: tf.keras.optimizers.Schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=1,
            decay_rate=self.learning_rate_decay,
            staircase=False)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        if self.predict_returns:
            self.loss: Union[str, Callable] = lambda y_actual, y_pred: custom_loss(y_actual, y_pred, 0.01)
            self.loss_name = 'combined'
        else:   
            self.loss = self.loss_name
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
        update_rolling_window: int = ArgsParser.get_or_default(params, 'update_rolling_window', 1)
        l2: float = ArgsParser.get_or_default(params, 'l2', 0.0)
        normalize_dict: Dict[str, Any] = ArgsParser.get_or_default(params, 'normalize', {})
        masked: bool = ArgsParser.get_or_default(params, 'masked', False)
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
        consume_returns: bool = ArgsParser.get_or_default(params, 'consume_returns', False)
        first_column_cash: bool = ArgsParser.get_or_default(params, 'first_column_cash', False)
        window_size: int = ArgsParser.get_or_default(params, 'window_size', 5)

        return ChalvatzisPredictor(input_size=input_size, output_size=output_size, window_size=window_size, hidden_size=hidden_size, 
            epochs=epochs, predict_returns=predict_returns, first_column_cash=first_column_cash, shuffle=shuffle, iter_per_item=iter_per_item, dropout=dropout, all_hidden=all_hidden, learning_rate_decay=learning_rate_decay,
            batch_size=batch_size, initial_learning_rate=initial_learning_rate, normalize=normalize, normalize_min=normalize_min, normalize_max=normalize_max,
            update_iter_per_item=update_iter_per_item, l2=l2, masked=masked, update_rolling_window=update_rolling_window, consume_returns=consume_returns)

    def prepare_prices(self, Y: np.ndarray) -> np.ndarray:
        partial = super().prepare_prices(Y)
        if not self.consume_returns:
            return partial
        else:
            return partial[1:, :].copy()

    def prepare_features(self, X: np.ndarray, predict=False) -> np.ndarray:
        if predict:
            if self.consume_returns:
                return (X[1:,:] / X[:-1,:] - 1).copy()
            else:
                return X.copy()
        else:
            if self.consume_returns:
                return (X[1:-2,:] / X[:-3,:] - 1).copy()
            else:
                return X[:-1,:].copy()

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

        s = X.shape[0]
        if self.consume_returns:
            s -=1
        if self.predict_returns:
            s -=1
            
        splits = [s]
        if X_val is not None and Y_val is not None:
            splits.append(X.shape[0] + X_val.shape[0])
            X = np.concatenate([X, X_val], axis=0)
            Y = np.concatenate([Y, Y_val], axis=0)
        
        X = self.prepare_features(X)
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
        
        verbose: int = ArgsParser.get_or_default(kwargs, 'verbose', 1)

        if (epochs_between_validation is not None):
            loops = np.ceil(self.epochs / float(epochs_between_validation))
            for loop in range(int(loops)):
                epochs: int = min(self.epochs, (1 + loop) * epochs_between_validation)
                obs = OutputObserver(self.model, X_train, Y_hat=Y_train * 0, Y_train=Y_train, epochs=epochs, initial_epoch=loop * epochs_between_validation, batch_size=self.batch_size, calculate_stats=lambda Y_hat, Y_train: self.calculate_stats(Y_hat, Y_train))
                self.model.fit(X_train, Y_train, epochs=epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[obs], initial_epoch=loop * epochs_between_validation, verbose=verbose)

                # Training Metrics
                training_mda = obs.mda
                training_corr = obs.corr
                if self.first_column_cash:
                    training_mda = np.concatenate([np.ones([training_mda.shape[0], 1]), training_mda], axis=1)
                    training_corr = np.concatenate([np.zeros([training_corr.shape[0], 1]), training_corr], axis=1)
                
                # Predict
                Y_hat = obs.Y_hat[::self.iter_per_item, -1, :]
                if self.normalize:
                    Y_hat = self.__normalize_apply_targets(Y_hat, revert=True)
                if (loop != loops - 1):
                    self.validate(loop, val_infra, Y_hat, epochs, training_mda=training_mda, training_corr=training_corr)
        else:
            obs = OutputObserver(self.model, X_train, Y_hat=Y_train * 0, Y_train=Y_train, epochs=self.epochs, initial_epoch=0, batch_size=self.batch_size, calculate_stats=lambda Y_hat, Y_train: self.calculate_stats(Y_hat, Y_train))
            self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, Y_val), callbacks=[obs], initial_epoch=0, verbose=verbose)
            # Predict
            Y_hat = obs.Y_hat[::self.iter_per_item, -1, :]
            if self.normalize:
                Y_hat = self.__normalize_apply_targets(Y_hat, revert=True)
        return Y_hat

    def calculate_stats(self, Y_hat, Y_train) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalize:
            Y_hat = self.__normalize_apply_targets(Y_hat, revert=True)
            Y_train = self.__normalize_apply_targets(Y_train, revert=True)
        if self.predict_returns:
            Y_hat = Y_hat[:,-1,:]
            Y_train = Y_train[:,-1,:]
        else:
            Y_hat = Y_hat[1:,-1,:] / Y_train[:-1,-1,:]
            Y_train = Y_train[1:,-1,:] / Y_train[:-1,-1,:]

        mda = np.array([np.mean((Y_hat[:,asset_i] * Y_train[:,asset_i] >= 0)) for asset_i in range(Y_hat.shape[1])])
        corr = np.array([np.corrcoef(Y_hat[:,asset_i], Y_train[:,asset_i])[0,1] for asset_i in range(Y_hat.shape[1])])
        return (mda, corr)

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
            X = self.prepare_features(X, predict=True)
            if self.normalize:
                X = self.__normalize_apply_features(X)
            data = self.__create_sequences(X, X, [X.shape[0]])
            X, _ = data[0]
            
            output = self.model.predict(X)[:,-1,:]
            if self.normalize:
                output = self.__normalize_apply_targets(output, revert=True)
            return output, np.abs(output)
        else:
            X = self.prepare_features(X[-self.window_size - 2:, :], predict=True)
            x = X[-self.window_size:, :]
            if self.normalize:
                x = self.__normalize_apply_features(x)
                
            output = self.model.predict(np.array([x]))[0,-1,:]
            if self.normalize:
                output = self.__normalize_apply_targets(output, revert=True)
            return output, np.abs(output)

    def _inner_update(self, X: np.ndarray, Y: np.ndarray) -> None:        
        if (self.update_rolling_window == 0) or (self.update_iter_per_item == 0):
            return

        x = self.prepare_features(X)
        y = self.prepare_prices(Y)
        x = x[-self.window_size + 1 - self.update_rolling_window:, :]
        y = y[-self.window_size + 1 - self.update_rolling_window:, :]

        if self.normalize:
            x = self.__normalize_apply_features(x)
            y = self.__normalize_apply_targets(y)

        data = self.__create_sequences(x, y, [x.shape[0]])
        x, y = data[0]
        
        X_train = np.array([x,] * self.update_iter_per_item).transpose([1,0,2,3]).reshape([x.shape[0] * self.update_iter_per_item] + list(x.shape[1:]))
        Y_train = np.array([y,] * self.update_iter_per_item).transpose([1,0,2,3]).reshape([y.shape[0] * self.update_iter_per_item] + list(y.shape[1:]))

        self.model.fit(X_train, Y_train, batch_size=self.batch_size, verbose=0)

    def detach_model(self) -> Any:
        m = self.model.detach_model()
        return m
    
    def copy_model(self) -> Any:
        model = LSTMChalvatzisTF(
            input_size=self.input_size, window_size=self.window_size, hidden_size=self.hidden_size, output_size=self.output_size,
            dropout=self.dropout, masked=self.masked)
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

    def validate(self, num, val_infra, Y_hat, last_epoch, training_mda, training_corr) -> None:
        print('Calculating validation within training...', end="\r")
        agent = copy.deepcopy(val_infra[0])
        market_execute = val_infra[1]
        timestamps = val_infra[2]
        instruments = val_infra[3]
        journal = copy.deepcopy(val_infra[4])
        train_num = val_infra[5]
        val_num = val_infra[6]
        X_train = val_infra[7].copy()
        Y_train = val_infra[8].copy()
        X_val = val_infra[9].copy()
        Y_val = val_infra[10].copy()
        trader = copy.deepcopy(val_infra[11])

        model_copy = self.copy_model()
        model = self.detach_model()
        predictor: Predictor = copy.deepcopy(self)
        self.attach_model(model)
        predictor.attach_model(model_copy)

        if self.first_column_cash:
            prediction, confidence = self.add_cash(Y_hat, Y_hat)

        trader.fit(prediction=prediction, conviction=prediction, Y=Y_train, predict_returns=predictor.predict_returns)

        agent.predictor = predictor
        agent.trader = trader
        X = np.concatenate([X_train, X_val], axis=0)
        Y = np.concatenate([Y_train, Y_val], axis=0)

        for i in range(val_num):
            idx = train_num + i
            timestamp = timestamps[idx]
            trade_orders, price_prediction = agent.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
            journal.store_order(trade_orders, price_prediction, timestamp)
            trade_fills = market_execute(trade_orders, timestamp)
            journal.store_fill(trade_fills)
            agent.update(trade_fills, X[:idx + 1, :], Y[:idx + 2, :])
            printed_string = 'Calculating validation within training... Progress: %.1f %%' % (100 * (i + 1) / val_num)
            print (printed_string, end="\r")
        
        print('Calculating validation within training... Progress: %.1f %% - Completed!' % (100 * (i + 1) / val_num))

        journal.run_analytics('train', timestamps[train_num:train_num + val_num], Y_val, instruments, name=num, last_epoch=last_epoch, 
            training_predictions=None, training_mda=training_mda, training_corr=training_corr)