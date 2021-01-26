from __future__ import annotations
from typing import Dict, Tuple
from abc import ABC, abstractclassmethod
from torch import Tensor


class Predictor(ABC):

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Predictor:
        raise NotImplementedError

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
