from __future__ import annotations
from typing import Dict, Tuple
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

    def fit(self, X: np.array, Y: np.array, **kwargs):
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y np.array by an index (a day).
        """
        X_tensor = Tensor(X)
        Y_tensor = Tensor(Y)
        if self.predict_returns:
            X_tensor = X_tensor[:-1,:]
            Y_tensor = Y_tensor[1:,:] / Y_tensor[:-1,:] - 1
        else:
            X_tensor = X_tensor[:-1,:]
            Y_tensor = Y_tensor[1:,:]
        if self.window_size == 1:
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = self.loss(outputs, Y_tensor)
                loss.backward()
                self.optimizer.step()
        
        else:
            X_tensor, Y_tensor = self.__reshape_window(X_tensor, Y_tensor)
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor[:-1,:])
                loss = self.loss(outputs, Y_tensor[1:,:])
                loss.backward()
                self.optimizer.step()
        
    def __reshape_window(self, X: Tensor, Y: Tensor):
        Y = Y[self.window_size-1:,:]
        X_new = empty(len(X)-self.window_size+1, self.window_size * len(X[0]))
        for i in range(len(X)-self.window_size+1):
            # In the case of linear predict, flatten
            X_new[i] = flatten(X[i:self.window_size+i])
        return X_new, Y

    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        """
        Returns:
            Tuple[np.array, np.array]: prediction and conviction
        """
        output = self.model(Tensor(X)[-1, :]).detach().numpy()
        return output, output * 1
