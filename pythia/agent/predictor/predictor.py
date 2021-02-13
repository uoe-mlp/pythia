from __future__ import annotations
from typing import Dict, Tuple, Optional
from abc import ABC, abstractclassmethod
import numpy as np


class Predictor(ABC):

    def __init__(self, input_size: int, output_size: int, predict_returns: bool):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.predict_returns: bool = predict_returns

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor: pass

    @abstractclassmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, **kwargs): pass

    @abstractclassmethod
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: pass # prediction and conviction

    def prepare_prices(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.predict_returns:
            X = X[:-1,:]
            Y = Y[1:,:] / Y[:-1,:] - 1
        else:
            X = X[:-1,:]
            Y = Y[1:,:]
        return (X, Y)