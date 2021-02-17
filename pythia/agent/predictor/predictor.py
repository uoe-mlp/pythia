from __future__ import annotations
from typing import Dict, Tuple, Optional
from abc import ABC, abstractclassmethod
import numpy as np


class Predictor(ABC):

    def __init__(self, input_size: int, output_size: int, predict_returns: bool, first_col_cash: bool):
        self.input_size: int = input_size
        self.output_size: int = output_size - 1 if first_col_cash else output_size
        self.first_col_cash = first_col_cash
        self.predict_returns: bool = predict_returns

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor: pass

    @abstractclassmethod
    def _inner_fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, **kwargs): pass

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, **kwargs):
        Y_new = self.remove_cash(Y.copy()) if self.first_col_cash else Y.copy()
        Y_val_new = self.remove_cash(Y_val.copy()) if self.first_col_cash and Y_val is not None else None
        self._inner_fit(X, Y_new, X_val, Y_val, **kwargs)

    @abstractclassmethod
    def _inner_predict(self, x: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]: pass
    
    def predict(self, x: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]: 
        prediction, confidence = self._inner_predict(x, all_history)

        if self.first_col_cash:
            prediction, confidence = self.add_cash(prediction, confidence)

        return prediction, confidence
        
    @abstractclassmethod
    def _inner_update(self, X: np.ndarray, Y: np.ndarray) -> None: pass

    def update(self, X: np.ndarray, Y: np.ndarray) -> None:
        Y_new = self.remove_cash(Y.copy()) if self.first_col_cash else Y.copy()
        self._inner_update(X, Y_new)

    def prepare_prices(self, Y: np.ndarray) -> np.ndarray:
        if self.predict_returns:
            Y = Y[1:,:] / Y[:-1,:] - 1
        else:
            Y = Y[1:,:]
        return Y

    def remove_cash(self, Y: np.ndarray) -> np.ndarray:
        Y = Y[:,1:]

    def add_cash(self, prediction: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        prediction_new = np.ones((prediction.shape[0],prediction.shape[1]+1))
        prediction_new[:,1:] = prediction

        confidence_new = np.ones((confidence.shape[0],confidence.shape[1]+1))
        confidence_new[:,1:] = confidence

        return prediction_new, confidence_new
