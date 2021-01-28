from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractstaticmethod
from torch import Tensor
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Market(ABC):

    def __init__(self, input_size: int, output_size: int, timestamps: List[Timestamp]):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.timestamps: List[Timestamp] = timestamps

    @staticmethod
    def initialise(params: Dict) -> Market:
        raise NotImplementedError

    @abstractclassmethod
    def execute(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        raise NotImplementedError

    @abstractclassmethod
    def simulate(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        raise NotImplementedError
