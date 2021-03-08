from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, List
from abc import ABC, abstractclassmethod
import numpy as np


class Predictor(ABC):

    def __init__(self, input_size: int, output_size: int, predict_returns: bool, first_column_cash: bool):
        self.input_size: int = input_size
        self.output_size: int = output_size - 1 if first_column_cash else output_size
        self.first_column_cash = first_column_cash
        self.predict_returns: bool = predict_returns

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor: pass

    @abstractclassmethod
    def _inner_fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, epochs_between_validation: Optional[int]=None, val_infra: Optional[List]=None, **kwargs): pass

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, epochs_between_validation: Optional[int]=None, val_infra: Optional[List]=None, **kwargs):
        Y_new = self.remove_cash(Y.copy()) if self.first_column_cash else Y.copy()
        Y_val_new = self.remove_cash(Y_val.copy()) if self.first_column_cash and Y_val is not None else None
        Y_hat = self._inner_fit(X, Y_new, X_val, Y_val_new, epochs_between_validation=epochs_between_validation, val_infra=val_infra, **kwargs)
        if self.first_column_cash:
            prediction, confidence = self.add_cash(Y_hat, Y_hat)
        else:
            prediction = Y_hat
        return prediction

    @abstractclassmethod
    def _inner_predict(self, x: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]: pass
    
    def predict(self, x: np.ndarray, all_history: bool=False) -> Tuple[np.ndarray, np.ndarray]: 
        prediction, confidence = self._inner_predict(x, all_history)

        if self.first_column_cash:
            prediction, confidence = self.add_cash(prediction, confidence)

        return prediction, confidence
        
    @abstractclassmethod
    def _inner_update(self, X: np.ndarray, Y: np.ndarray) -> None: pass

    def update(self, X: np.ndarray, Y: np.ndarray) -> None:
        Y_new = self.remove_cash(Y.copy()) if self.first_column_cash else Y.copy()
        self._inner_update(X, Y_new)

    def prepare_prices(self, Y: np.ndarray) -> np.ndarray:
        if self.predict_returns:
            return Y[1:,:] / Y[:-1,:] - 1
        else:
            return Y[1:,:]

    def remove_cash(self, Y: np.ndarray) -> np.ndarray:
        return Y[:,1:]

    def add_cash(self, prediction: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        if len(prediction.shape) == 2:
            if self.predict_returns:
                prediction_new = np.zeros((prediction.shape[0],prediction.shape[1]+1))
                confidence_new = np.zeros((confidence.shape[0],confidence.shape[1]+1))
            else:
                prediction_new = np.ones((prediction.shape[0],prediction.shape[1]+1))
                confidence_new = np.ones((confidence.shape[0],confidence.shape[1]+1))

            prediction_new[:,1:] = prediction
            confidence_new[:,1:] = confidence
        else:
            if self.predict_returns:
                prediction_new = np.zeros((prediction.shape[0]+1,))
                confidence_new = np.zeros((confidence.shape[0]+1,))
            else:
                prediction_new = np.ones((prediction.shape[0]+1,))
                confidence_new = np.ones((confidence.shape[0]+1,))

            prediction_new[1:] = prediction
            confidence_new[1:] = confidence
        return prediction_new, confidence_new

    def detach_model(self): return None

    def attach_model(self, model): pass

    def copy_model(self): return None