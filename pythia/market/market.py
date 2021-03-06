from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractstaticmethod
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Market(ABC):

    def __init__(self, input_size: int, output_size: int, timestamps: List[Timestamp], instruments: List[str]):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.timestamps: List[Timestamp] = timestamps
        self.instruments: List[str] = instruments

    @staticmethod
    def initialise(params: Dict) -> Market: pass

    @abstractclassmethod
    def execute(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]: pass

    @abstractclassmethod
    def simulate(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]: pass
