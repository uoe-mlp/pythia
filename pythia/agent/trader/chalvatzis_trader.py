from __future__ import annotations
from typing import Dict, List, Optional
from abc import ABC, abstractclassmethod, abstractproperty
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrderSell, TradeOrderBuy, TradeOrder
from pythia.journal import TradeFill
from pythia.utils import ArgsParser

from .trader import Trader


class ChalvatzisTrader(Trader):

    def __init__(self, output_size: int, first_target_cash: bool):
        super(ChalvatzisTrader, self).__init__(output_size=output_size)
        self.first_target_cash: bool = first_target_cash
        self.expected_returns: np.ndarray = np.array([])
        self.realised_returns: np.ndarray = np.array([])
        self.bins: np.ndarray = np.array([])
        self.quantiles: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    @staticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        first_target_cash: bool = ArgsParser.get_or_default(params, 'first_target_cash', True)
        return ChalvatzisTrader(output_size, first_target_cash)

    def fit(self, prediction: np.ndarray, conviction: np.ndarray, Y: np.ndarray, predict_returns: bool, **kwargs):
        if not predict_returns:
            previous_prices = Y[:-1, :]
            prediction = prediction / previous_prices[-prediction.shape[0]:, :] - 1

        expected_returns = prediction
        realised_returns = Y[1:, :] / Y[:-1, :] - 1

        max_common_size = min([expected_returns.shape[0], realised_returns.shape[0]])

        self.expected_returns = expected_returns[-max_common_size:, :]
        self.realised_returns = realised_returns[-max_common_size:, :]

        if self.first_target_cash:
            self.expected_returns = self.expected_returns[:, 1:]
            self.realised_returns = self.realised_returns[:, 1:]

        self.__update_bins()

    def act(self, prediction: np.ndarray, conviction: np.ndarray, timestamp: Timestamp, prices: np.ndarray, predict_returns: bool) -> List[TradeOrder]:
        if predict_returns:
            target_trade = int(np.argmax(prediction))
        else:
            target_trade = int(np.argmax(prediction / prices))

        target_portfolio: np.ndarray = self._portfolio * 0
        target_portfolio[target_trade] = 1
        current_portfolio: np.ndarray = self.portfolio * prices
        current_portfolio /= sum(current_portfolio)
        delta = target_portfolio - current_portfolio
        
        trades: List[TradeOrder] = [TradeOrderSell(i, timestamp, x) for i, x in enumerate(self._portfolio) if float(delta[i]) < 0]
        trades += [TradeOrderBuy(target_trade, timestamp, percentage=1) for i, x in enumerate(self._portfolio) if float(delta[i]) > 0]
        return trades
        
    def __update_bins(self) -> None:
        self.bins = np.concatenate([
            np.zeros((1, self.expected_returns.shape[1],)),
            np.quantile(np.abs(self.expected_returns), self.quantiles, axis=0),
            np.ones((1, self.expected_returns.shape[1],)) * np.inf],
            axis=0)

        for col in range(self.expected_returns.shape[1]):
            for row in range(self.expected_returns.shape[0]):
                exp_ret = self.expected_returns[row:, col]
                real_ret = self.realised_returns[row:, col]
                if exp_ret[0] >= 0:
                    first_negative = (exp_ret < 0).argmax(axis=0)
                    real_ret[:first_negative]

                    raise ValueError() # TODO: continue from here
                