from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
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

    def fit(self, X: Tensor, Y: Tensor, **kwargs):
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y Tensor by an index (a day).
        """
        if self.predict_returns:
            X = X[:-1,:]
            Y = Y[1:,:] / Y[:-1,:] - 1
        else:
            X = X[:-1,:]
            Y = Y[1:,:]
        if self.window_size == 1:
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.loss(outputs, Y)
                loss.backward()
                self.optimizer.step()
        
        else:
            X, Y = self.__reshape_window(X, Y)
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X[:-1,:])
                loss = self.loss(outputs, Y[1:,:])
                loss.backward()
                self.optimizer.step()
        
    def __reshape_window(self, X: Tensor, Y: Tensor):
        Y = Y[self.window_size-1:,:]
        X_new = empty(len(X)-self.window_size+1, self.window_size * len(X[0]))
        for i in range(len(X)-self.window_size+1):
            # In the case of linear predict, flatten
            X_new[i] = flatten(X[i:self.window_size+i])
        return X_new, Y

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            Tuple[Tensor, Tensor]: prediction and conviction
        """
        return self.model(X[-1, :]), self.model(X[-1, :]) * 1
