from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
import numpy as np


class Predictor(ABC):

    def __init__(self, input_size: int, output_size: int, predict_returns: bool):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.predict_returns: bool = predict_returns

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, X: np.array, Y: np.array, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, x: np.array) -> Tuple[np.array, np.array]: # prediction and conviction
        raise NotImplementedError
