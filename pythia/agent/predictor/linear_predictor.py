from __future__ import annotations
from typing import Dict, Tuple, Optional
from abc import ABC, abstractclassmethod
import numpy as np
from torch import Tensor, empty, flatten
from torch.optim import Adam, Optimizer
from torch.nn import MSELoss, Module

from pythia.utils import ArgsParser
from pythia.agent.network import LinearRegression

from .predictor import Predictor


class LinearPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, window_size: int, learning_rate: float, weight_decay: float, epochs: int, predict_returns: bool):
        super(LinearPredictor, self).__init__(input_size, output_size, predict_returns)
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.model: LinearRegression = LinearRegression(input_size, output_size)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss: Module = MSELoss()
        self.epochs: int = epochs
        self.window_size: int = window_size

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        weight_decay: float = ArgsParser.get_or_default(params, 'weight_decay', 0.0)
        learning_rate: float = ArgsParser.get_or_default(params, 'learning_rate', 1e-3)
        epochs: int = ArgsParser.get_or_default(params, 'epochs', 100)
        predict_returns: bool = ArgsParser.get_or_default(params, 'predict_returns', False)
        window_size: int = ArgsParser.get_or_default(params, 'window_size', 1)
        return LinearPredictor(input_size=input_size, output_size=output_size, window_size=window_size, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs, predict_returns=predict_returns)

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y np.ndarray by an index (a day).
        """
        X = X[:-1,:]
        Y = self.prepare_prices(Y)
        X_tensor = Tensor(X)
        Y_tensor = Tensor(Y)

        X_tensor, Y_tensor = self.__reshape_window(X_tensor, Y_tensor)
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.loss(outputs, Y_tensor)
            loss.backward()
            self.optimizer.step()
        
        return self.model(Tensor(X_tensor)).detach().numpy()
        
    def __reshape_window(self, X: Tensor, Y: Optional[Tensor]):
        if Y is not None:
            Y = Y[self.window_size-1:,:]
        X_new = empty(len(X)-self.window_size+1, self.window_size * len(X[0]))
        for i in range(len(X)-self.window_size+1):
            # In the case of linear predict, flatten
            X_new[i] = flatten(X[i:self.window_size+i])
        return X_new, Y

    def predict(self, X: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: prediction and conviction
        """
        if all_history:
            X_tensor = Tensor(X)
            X_tensor, _ = self.__reshape_window(X_tensor, None)
            output = self.model(Tensor(X))
            return output.detach().numpy(), output.detach().numpy() * 1
        else:
            X = X[-self.window_size:, :]
            X_tensor = Tensor(X)
            X_tensor, _ = self.__reshape_window(X_tensor, None)

            output = self.model(Tensor(X)[-1, :]).detach().numpy()
            return output, output * 1

    def update(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = X[-self.window_size:, :]
        Y = Y[-self.window_size-1:, :]
        Y = self.prepare_prices(Y)
        Y = Y[-X.shape[0]:, :]
        X_tensor = Tensor(X)
        Y_tensor = Tensor(Y)
        X_tensor, Y_tensor = self.__reshape_window(X_tensor, Y_tensor)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor[-1:,:])
            loss = self.loss(outputs, Y_tensor[-1:,:])
            loss.backward()
            self.optimizer.step()