from __future__ import annotations
from abc import abstractmethod
import os
from typing import Optional, List, Dict, cast
import copy
import numpy as np
from pythia import market
from pythia import journal

from pythia.utils import ArgsParser
from pythia.journal import Journal
from pythia.market import Market
from pythia.agent import Agent
from pythia.journal.trade_fill import TradeFill
from pythia.journal.trade_order import TradeOrder

from .experiment import Experiment


class StandardExperiment(Experiment):

    DEFAULT_SPLIT: List[float] = [0.7, 0.15, 0.15]

    def __init__(self, path: str, market: Market, agent: Agent, journal: Journal, benchmark: Optional[Agent], settings: Dict, 
    train: float, val: float, test: float, epochs_between_validation: Optional[int], retrain_before_test: bool):
        super(StandardExperiment, self).__init__(path, market, agent, journal, benchmark, settings)
        self.train: float = train
        self.val: float = val
        self.test: float = test
        self.epochs_between_validation: Optional[int] = epochs_between_validation
        self.retrain_before_test: bool = retrain_before_test

    @staticmethod
    def initialise(path: str, market: Market, agent: Agent, journal: Journal, benchmark: Optional[Agent], settings: Dict, params: Dict=None) -> Experiment:
        p: Dict = params if params is not None else {}
        train: Optional[float] = ArgsParser.get_or_default(p, 'train', None)
        val: Optional[float] = ArgsParser.get_or_default(p, 'val', None)
        test: Optional[float] = ArgsParser.get_or_default(p, 'test', None)
        epochs_between_validation: Optional[int] = ArgsParser.get_or_default(p, 'epochs_between_validation', None)
        retrain_before_test: bool = ArgsParser.get_or_default(p, 'retrain_before_test', False)

        fractions: List[Optional[float]] = [train, val, test]

        available = sum([x for x in fractions if x is not None])
        if available > 1:
            fractions = [x / available if x is not None else None for x in fractions]
            
        if not any(fractions):
            # If non available, use default
            clean_fractions: List[float] = StandardExperiment.DEFAULT_SPLIT
        elif not all(fractions):
            # If some available, fill missing following default proportions
            missing = 1 - available
            missing_defaults = sum([x for x, y in zip(StandardExperiment.DEFAULT_SPLIT, fractions) if y is None])
            clean_fractions = [x if x is not None else y * missing / missing_defaults for x, y in zip(fractions, StandardExperiment.DEFAULT_SPLIT)]
        else:
            # If all available, just use them. Had to use cast here cause linter could not pick the logic up. 
            clean_fractions = cast(List[float], fractions) 

        return StandardExperiment(path, market, agent, journal, benchmark, settings, train=clean_fractions[0], val=clean_fractions[1], test=clean_fractions[2],
            epochs_between_validation=epochs_between_validation, retrain_before_test=retrain_before_test)

    def run(self):
        X = self.market.X
        Y = self.market.Y

        train_num = round(X.shape[0] * self.train)
        val_num = round(X.shape[0] * self.val)
        test_num = X.shape[0] - train_num - val_num

        X_train = X[0:train_num, :]
        Y_train = Y[0:train_num, :]
        X_val = X[train_num:train_num+val_num, :]
        Y_val = Y[train_num:train_num+val_num, :]
        X_test = X[train_num+val_num:, :]
        Y_test = Y[train_num+val_num:, :]

        if self.epochs_between_validation:
            val_infra = [
                copy.deepcopy(self.agent),
                self.market.execute,
                self.market.timestamps,
                self.market.instruments,
                copy.deepcopy(self.journal),
                train_num,
                val_num,
                X_train,
                Y_train,
                X_val,
                Y_val,
            ]
        else:
            val_infra = []

        self.agent.fit(X_train, Y_train, X_val, Y_val, 
            simulator=lambda orders, timestamp: self.market.simulate(orders, timestamp) 
            if timestamp <= self.market.timestamps[train_num + val_num - 1] 
            else ValueError('Date is out of traning or validation period.'),
            epochs_between_validation=self.epochs_between_validation,
            val_infra=val_infra)

        if self.epochs_between_validation:
            self.journal.compile_results(self.market.instruments)

        if self.benchmark:
            self.benchmark.fit(X_train, Y_train, X_val, Y_val, 
                simulator=lambda orders, timestamp: self.market.simulate(orders, timestamp) 
                if timestamp <= self.market.timestamps[train_num + val_num - 1] 
                else ValueError('Date is out of traning or validation period.'))

        trade_orders_benchmark: Optional[List[TradeOrder]] = None
        trade_fills_benchmark: Optional[List[TradeFill]] = None

        print('Calculating validation...', end="\r")
        for i in range(val_num):
            idx = train_num + i
            timestamp = self.market.timestamps[idx]
            trade_orders, price_prediction = self.agent.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
            self.journal.store_order(trade_orders, price_prediction, timestamp)
            trade_fills = self.market.execute(trade_orders, timestamp)
            self.journal.store_fill(trade_fills)
            self.agent.update(trade_fills, X[:idx + 1, :], Y[:idx + 2, :])
            
            if self.benchmark:
                trade_orders_benchmark, _ = self.benchmark.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
                self.benchmark_journal.store_order(trade_orders_benchmark)
                trade_fills_benchmark = self.market.execute(trade_orders_benchmark, timestamp)
                self.benchmark_journal.store_fill(trade_fills_benchmark)
                self.benchmark.update(trade_fills_benchmark, X[:idx + 1, :], Y[:idx + 2, :])
            
            printed_string = 'Calculating validation... Progress: %.1f %%' % (100 * (i + 1) / val_num)
            print (printed_string, end="\r")

        print('Calculating validation... Progress: %.1f %% - Completed!' % (100 * (i + 1) / val_num))

        self.journal.run_analytics('validation', self.market.timestamps[train_num:train_num + val_num], Y_val, self.market.instruments)
        self.journal.clean()
        self.agent.clean_portfolio()
        if self.benchmark:
            self.benchmark_journal.run_analytics(os.path.join('validation', 'benchmark'), self.market.timestamps[train_num:train_num + val_num], Y_val, self.market.instruments)
            self.benchmark_journal.clean()
            self.benchmark.clean_portfolio()
        
        if self.retrain_before_test:
            print('Retraining before test...')        
            
            X_non_test = X[:train_num+val_num, :]
            Y_non_test = Y[:train_num+val_num, :]

            self.agent.fit(X_non_test, Y_non_test, X_non_test, Y_non_test, 
            simulator=lambda orders, timestamp: self.market.simulate(orders, timestamp) 
            if timestamp <= self.market.timestamps[train_num + val_num - 1] 
            else ValueError('Date is out of traning or validation period.'),
            epochs_between_validation=None)

        print('Calculating test...', end="\r")
        for i in range(test_num):
            idx = train_num + val_num + i
            timestamp = self.market.timestamps[idx]
            trade_orders, price_prediction = self.agent.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
            self.journal.store_order(trade_orders, price_prediction, timestamp)
            trade_fills = self.market.execute(trade_orders, timestamp)
            self.journal.store_fill(trade_fills)
            self.agent.update(trade_fills, X[:idx + 1, :], Y[:idx + 2, :])
            
            if self.benchmark:
                trade_orders_benchmark, _ = self.benchmark.act(X[:idx + 1, :], timestamp, Y[:idx + 1, :])
                self.benchmark_journal.store_order(trade_orders_benchmark)
                trade_fills_benchmark = self.market.execute(trade_orders_benchmark, timestamp)
                self.benchmark_journal.store_fill(trade_fills_benchmark)
                self.benchmark.update(trade_fills_benchmark, X[:idx + 1, :], Y[:idx + 2, :])
            
            printed_string = 'Calculating test... Progress: %.1f %%' % (100 * (i + 1) / test_num)
            print (printed_string, end="\r")
        
        print('Calculating test... Progress: %.1f %% - Completed!' % (100 * (i + 1) / test_num))

        self.journal.run_analytics('test', self.market.timestamps[train_num + val_num:], Y_test, self.market.instruments)
        if self.benchmark:
            self.benchmark_journal.run_analytics(os.path.join('test', 'benchmark'), self.market.timestamps[train_num + val_num:], Y_test, self.market.instruments)
        self.journal.export_settings(self.settings)
