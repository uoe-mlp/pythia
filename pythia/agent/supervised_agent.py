from __future__ import annotations
from typing import Callable, Dict, Optional, List
from abc import ABC, abstractclassmethod
import numpy as np
from pandas import Timestamp

from pythia.utils import ArgsParser
from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .agent import Agent
from .predictor import Predictor, LinearPredictor, ChalvatzisPredictor, FlatPredictor
from .trader import Trader, NaiveTrader, BuyAndHoldTrader, ChalvatzisTrader


class SupervisedAgent(Agent):

    def __init__(self, predictor: Predictor, trader: Trader):
        self.predictor: Predictor = predictor
        self.trader: Trader = trader

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> SupervisedAgent:
        # ---- PREDICTOR ----
        predictor_config = ArgsParser.get_or_default(params, 'predictor', {'type': 'linear'})
        predictor_params = ArgsParser.get_or_default(predictor_config, 'params', {})
        if predictor_config['type'].lower() == 'linear':
            predictor: Predictor = LinearPredictor.initialise(input_size=input_size, output_size=output_size, params=predictor_params)
        elif predictor_config['type'].lower() == 'chalvatzis':
            predictor = ChalvatzisPredictor.initialise(input_size=input_size, output_size=output_size, params=predictor_params)
        elif predictor_config['type'].lower() == 'flat':
            predictor = FlatPredictor.initialise(input_size=input_size, output_size=output_size, params=predictor_params)
        else:
            raise ValueError('Predictor type "%s" not recognized'  % (predictor_config['type']))

        # ---- TRADER ----
        trader_config = ArgsParser.get_or_default(params, 'trader', {'type': 'naive'})
        trader_params = ArgsParser.get_or_default(trader_config, 'params', {})

        if trader_config['type'].lower() == 'naive':
            trader: Trader = NaiveTrader.initialise(output_size=output_size, params=trader_params)
        elif trader_config['type'].lower() == 'buy_and_hold':
            trader = BuyAndHoldTrader.initialise(output_size=output_size, params=trader_params)
        elif trader_config['type'].lower() == 'chalvatzis':
            trader = ChalvatzisTrader.initialise(output_size=output_size, params=trader_params)
        else:
            raise ValueError('Trader type "%s" not recognized'  % (trader_config['type']))

        return SupervisedAgent(predictor, trader)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, simulator: Callable[[List[TradeOrder], Timestamp], List[TradeFill]], **kwargs):
        prediction = self.predictor.fit(X_train, Y_train, X_val, Y_val, **kwargs)
        self.trader.fit(prediction=prediction, conviction=prediction, Y=Y_train, predict_returns=self.predictor.predict_returns)

    def act(self, X: np.ndarray, timestamp: Timestamp, Y: np.ndarray) -> List[TradeOrder]:
        prediction, conviction = self.predictor.predict(X)
        return self.trader.act(prediction, conviction, timestamp, prices=Y[-1, :], predict_returns=self.predictor.predict_returns)

    def update(self, fills: List[TradeFill], X: np.ndarray, Y: np.ndarray):
        self.trader.update_portfolio(fills)
        self.predictor.update(X, Y)
        prediction, conviction = self.predictor.predict(X)
        self.trader.update_policy(X, Y, prediction, conviction)

    def clean_portfolio(self) -> None:
        return self.trader.clean_portfolio()