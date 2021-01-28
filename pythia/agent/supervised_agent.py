from __future__ import annotations
from typing import Callable, Dict, Optional, List
from abc import ABC, abstractclassmethod
from torch import Tensor
from pandas import Timestamp

from pythia.utils import ArgsParser
from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .agent import Agent
from .predictor import Predictor, LinearPredictor
from .trader import Trader, NaiveTrader


class SupervisedAgent(Agent):

    def __init__(self, predictor: Predictor, trader: Trader):
        self.predictor: Predictor = predictor
        self.trader: Trader = trader

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> SupervisedAgent:
        # ---- PREDICTOR ----
        predictor_config = ArgsParser.get_or_default(params, 'predictor', {'type': 'linenar'})
        predictor_params = ArgsParser.get_or_default(predictor_config, 'params', {})

        if predictor_config['type'].lower() == 'linear':
            predictor: Predictor = LinearPredictor.initialise(input_size=input_size, output_size=output_size, params=predictor_params)
        else:
            raise ValueError('Predictor type "%s" not recognized'  % (predictor_config['type']))

        # ---- TRADER ----
        trader_config = ArgsParser.get_or_default(params, 'trader', {'type': 'naive'})
        trader_params = ArgsParser.get_or_default(trader_config, 'params', {})

        if trader_config['type'].lower() == 'naive':
            trader: Trader = NaiveTrader.initialise(output_size=output_size, params=trader_params)
        else:
            raise ValueError('Trader type "%s" not recognized'  % (trader_config['type']))

        return SupervisedAgent(predictor, trader)

    def fit(self, X_train: Tensor, Y_train: Tensor, X_val: Tensor, Y_val: Tensor, simulator: Callable[[List[TradeOrder], Timestamp], List[TradeFill]], **kwargs):
        self.predictor.fit(X_train, Y_train, **kwargs)
        prediction, conviction = self.predictor.predict(X_train)
        self.trader.fit(prediction=prediction, conviction=conviction, Y=Y_train)

    def act(self, x: Tensor, timestamp: Timestamp) -> List[TradeOrder]:
        prediction, conviction = self.predictor.predict(x)
        return self.trader.act(prediction, conviction, timestamp)

    def update_portfolio(self, fills: List[TradeFill]):
        self.trader.update_portfolio(fills)