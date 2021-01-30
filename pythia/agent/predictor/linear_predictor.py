from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.nn import MSELoss, Module

from pythia.utils import ArgsParser
from pythia.agent.network import LinearRegression

from .predictor import Predictor


class LinearPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, learning_rate: float, weight_decay: float, epochs: int):
        super(LinearPredictor, self).__init__(input_size, output_size, False)
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.model: LinearRegression = LinearRegression(input_size, output_size)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss: Module = MSELoss()
        self.epochs: int = epochs

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        weight_decay: float = ArgsParser.get_or_default(params, 'weight_decay', 0.0)
        learning_rate: float = ArgsParser.get_or_default(params, 'learning_rate', 1e-3)
        epochs: int = ArgsParser.get_or_default(params, 'epochs', 100)
        return LinearPredictor(input_size=input_size, output_size=output_size, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)

    def fit(self, X: Tensor, y: Tensor, **kwargs):
        lag: int = ArgsParser.get_or_default(kwargs, 'lag', 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X[:-lag,:])
            loss = self.loss(outputs, y[lag:,:])
            loss.backward()
            self.optimizer.step()

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            Tuple[Tensor, Tensor]: prediction and conviction
        """
        return self.model(x), self.model(x) * 1
