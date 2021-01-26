from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
from torch import Tensor
from torch.optim import Adam, Optimizer

from pythia.utils import ArgsParser
from pythia.agent.network import LinearRegression

from .predictor import Predictor


class LinearPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int, learning_rate: float, weight_decay: float):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.model: LinearRegression = LinearRegression(input_size, output_size)
        self.optimizer: Optimizer = Adam(self.model.linear, lr=learning_rate, weight_decay=weight_decay)

    @staticmethod
    def initialise(params: Dict) -> Predictor:
        weight_decay: float = ArgsParser.get_or_default(params, 'weight_decay', 0.0)
        learning_rate: float = ArgsParser.get_or_default(params, 'learning_rate', 1e-3)
        input_size: int = ArgsParser.get_or_error(params, 'input_size')
        output_size: int = ArgsParser.get_or_error(params, 'output_size')
        return LinearPredictor(input_size=input_size, output_size=output_size, learning_rate=learning_rate, weight_decay=weight_decay)

    @abstractclassmethod
    def fit(self, X: Tensor, y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """[summary]

        Args:
            X (Tensor): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Tuple[Tensor, Tensor]: prediction and conviction
        """
        raise NotImplementedError
