from __future__ import annotations
from typing import Callable, Dict, Optional, List, Any, Tuple
from abc import ABC, abstractclassmethod
import numpy as np
from pandas import Timestamp
import copy

from pythia.utils import ArgsParser
from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .agent import Agent
from .predictor import Predictor, LinearPredictor, ChalvatzisPredictor, FlatPredictor
from .trader import Trader, NaiveTrader, BuyAndHoldTrader, ChalvatzisTrader


class SupervisedAgent(Agent):

    def __init__(self, predictor: Predictor, trader: Trader, retrain_every: Optional[int]=None):
        self.predictor: Predictor = predictor
        self.trader: Trader = trader
        self.retrain_every: Optional[int] = retrain_every
        if self.retrain_every is not None:
            self.retrain_counter: int = 0

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> SupervisedAgent:
        retrain_every: Optional[int] = ArgsParser.get_or_default(params, 'retrain_every', None)
        # ---- PREDICTOR ----
        predictor_config = ArgsParser.get_or_default(params, 'predictor', {'type': 'flat'})
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
        trader_config = ArgsParser.get_or_default(params, 'trader', {'type': 'buy_and_hold'})
        trader_params = ArgsParser.get_or_default(trader_config, 'params', {})

        if trader_config['type'].lower() == 'naive':
            trader: Trader = NaiveTrader.initialise(output_size=output_size, params=trader_params)
        elif trader_config['type'].lower() == 'buy_and_hold':
            trader = BuyAndHoldTrader.initialise(output_size=output_size, params=trader_params)
        elif trader_config['type'].lower() == 'chalvatzis':
            trader = ChalvatzisTrader.initialise(output_size=output_size, params=trader_params)
        else:
            raise ValueError('Trader type "%s" not recognized'  % (trader_config['type']))

        return SupervisedAgent(predictor, trader, retrain_every=retrain_every)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: Optional[np.ndarray]=None, Y_val: Optional[np.ndarray]=None, 
            simulator: Optional[Callable[[List[TradeOrder], Timestamp], List[TradeFill]]]=None, epochs_between_validation: Optional[int]=None, val_infra: Optional[List]=None, **kwargs):
        prediction = self.predictor.fit(X_train, Y_train, X_val, Y_val, epochs_between_validation=epochs_between_validation, val_infra=None if val_infra is None else val_infra + [copy.deepcopy(self.trader)], **kwargs)
        self.trader.fit(prediction=prediction, conviction=prediction, Y=Y_train, predict_returns=self.predictor.predict_returns)

    def act(self, X: np.ndarray, timestamp: Timestamp, Y: np.ndarray) -> Tuple[List[TradeOrder], Optional[np.ndarray]]:
        prediction, conviction = self.predictor.predict(X)
        if self.predictor.predict_returns:
            price_prediction = (prediction + 1) * Y[-1, :]
        else:
            price_prediction = prediction
        return (self.trader.act(prediction, conviction, timestamp, prices=Y[-1, :], predict_returns=self.predictor.predict_returns), price_prediction)

    def update(self, fills: List[TradeFill], X: np.ndarray, Y: np.ndarray):
        self.trader.update_portfolio(fills)
        
        if self.retrain_every is not None:
            if self.retrain_counter == self.retrain_every:
                self.fit(X, Y, verbose=0)
                self.retrain_counter = 0
            else:
                self.retrain_counter += 1
                self.predictor.update(X, Y)
        else:
            self.predictor.update(X, Y)
                
        prediction, conviction = self.predictor.predict(X)
        self.trader.update_policy(X, Y, prediction, conviction, predict_returns=self.predictor.predict_returns)

    def clean_portfolio(self) -> None:
        return self.trader.clean_portfolio()
