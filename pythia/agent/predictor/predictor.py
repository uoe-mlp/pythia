from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
from torch import Tensor


class Predictor(ABC):

    def __init__(self, input_size: int, output_size: int, predicting_returns: bool):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.predicting_returns: bool = predicting_returns

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, X: Tensor, y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]: # prediction and conviction
        raise NotImplementedError
