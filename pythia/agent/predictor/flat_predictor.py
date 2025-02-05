from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractclassmethod
import numpy as np
from torch import Tensor, empty, flatten
from torch.optim import Adam, Optimizer
from torch.nn import MSELoss, Module

from pythia.utils import ArgsParser
from pythia.agent.network import LinearRegression

from .predictor import Predictor


class FlatPredictor(Predictor):

    def __init__(self, input_size: int, output_size: int):
        super(FlatPredictor, self).__init__(input_size, output_size, predict_returns=True, first_column_cash=False)

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        return FlatPredictor(input_size=input_size, output_size=output_size)

    def _inner_fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, epochs_between_validation: Optional[int]=None, val_infra: Optional[List]=None, **kwargs) -> np.ndarray:
        """
        Description:
            The X and Y tensors are data representative of the same day.
            Since the aim is to predict next day price, we need to lag
            the Y np.ndarray by an index (a day).
        """
        if self.predict_returns:
            return Y * 0
        else:
            return Y * 1
        
    def _inner_predict(self, X: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: prediction and conviction
        """
        if all_history:
            return np.zeros((X.shape[0], self.output_size)), np.ones((X.shape[0], self.output_size))
        else:
            return np.zeros((1, self.output_size)), np.ones((1, self.output_size))

    def _inner_update(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass