from os.path import isdir
from typing import List, Optional, Tuple, Dict
import os
from pandas._libs.tslibs import Timestamp
import numpy as np
import pandas as pd
from datetime import datetime
import json
import copy
import re
from pythia.utils import ArgsParser
import csv

from .trade_order import TradeOrder
from .trade_fill import TradeFill
from .analytics import Analytics


class Journal(object):

    def __init__(self, experiment_folder: str):
        self.experiment_folder: str = experiment_folder
        self.open_orders: List[TradeOrder] = []
        self.trades: List[Tuple[TradeOrder, TradeFill]] = []
        self.analytics: Optional[Analytics] = None
        self.started_at: datetime = datetime.now()
        self.timestamp: str = self.started_at.strftime('%Y%m%d_%H%M%S')
        self.predictions: Dict[Timestamp, np.ndarray] = {}

    def store_order(self, orders: List[TradeOrder], price_prediction: Optional[np.ndarray]=None, price_prediction_timestamp: Optional[Timestamp]=None):
        self.open_orders += orders
        if price_prediction is not None:
            self.predictions[price_prediction_timestamp] = price_prediction

    def store_fill(self, fills: List[TradeFill]):
        for fill in fills:
            same_id = [x for x in self.open_orders if x.id == fill.id]
            if len(same_id) == 1:
                self.trades.append((same_id[0], fill))
                self.open_orders = [x for x in self.open_orders if x.id != fill.id]
            else:
                raise ValueError('One and only one open order should match the id for this fill.')

    def run_analytics(self, type: str, timestamps: List[Timestamp], prices: np.ndarray, instruments: List[str], name: Optional[str]=None, training_predictions: Optional[pd.DataFrame]=None, **kwargs):
        self.analytics = Analytics.initialise(timestamps, [x[1] for x in self.trades], prices, self.predictions, instruments, training_predictions=training_predictions)
        analytics = self.analytics.to_dict()
        analytics['fills'] = sum([[{
            'direction': x.direction,
            'instrument': instruments[x.instrument],
            'price': x.price,
            'quantity': x.quantity,
            'value': x.value,
            'completed': x.completed.isoformat(),
        } for x in trade if isinstance(x, TradeFill)] for trade in self.trades],[])
        analytics['number_of_trades'] = len(analytics['fills'])
        analytics.update(**kwargs)

        if not os.path.isdir(self.experiment_folder):
            os.mkdir(self.experiment_folder)

        timestamp_folder = os.path.join(self.experiment_folder, 'results__' + self.timestamp)
        if not os.path.isdir(timestamp_folder):
            os.mkdir(timestamp_folder)

        type_folder = os.path.join(timestamp_folder, type)
        if not os.path.isdir(type_folder):
            os.mkdir(type_folder)

        if name is not None:
            filename = '%s.json' % (name)
        else:
            filename = 'data.json'

        with open(os.path.join(type_folder, filename), 'w') as fp:
            json.dump(analytics, fp, indent=4, sort_keys=True)

    def compile_results(self, instruments: List[str]):
        train_folder = os.path.join(self.experiment_folder, 'results__' + self.timestamp, 'train')

        all_metrics: List = ["epochs", "cum_return", "max_drawdown",
                            "n_trades", "sharpe", "sortino", "volatility"] + \
                            ["%s_mda" % instr for instr in instruments] + \
                            ["%s_correlation" % instr for instr in instruments]
        
        # Since the files are processed out of order keep a dictionary
        ordered_metrics: Dict = {}

        # Get list of all files
        if isdir(train_folder):
            train_reports = [f for f in os.listdir(train_folder)]
            for i, report in enumerate(train_reports):
                subdir = os.path.join(train_folder, report)
                with open(subdir, 'r') as fp:
                    data: Dict = json.load(fp)

                last_epoch = ArgsParser.get_or_default(data, 'last_epoch', None)
                cumulative_return = ArgsParser.get_or_default(data, 'cumulative_return', None)
                maximum_drawdown = ArgsParser.get_or_default(data, 'maximum_drawdown', None)
                number_of_trades = ArgsParser.get_or_default(data, 'number_of_trades', None)
                sharpe_ratio = ArgsParser.get_or_default(data, 'sharpe_ratio', None)
                sortino_ratio = ArgsParser.get_or_default(data, 'sortino_ratio', None)
                volatility = ArgsParser.get_or_default(data, 'volatility', None)
                mean_directional_accuracy = ArgsParser.get_or_default(data, 'mean_directional_accuracy', [None for instr in instruments])
                correlation = ArgsParser.get_or_default(data, 'correlation', [None for instr in instruments])

                ordered_metrics[i] = [last_epoch, cumulative_return, maximum_drawdown, number_of_trades,
                                    sharpe_ratio, sortino_ratio, volatility] + mean_directional_accuracy + correlation


            output_csv = os.path.join(train_folder, "output.csv")
            # Write to csv
            with open(output_csv, 'w+', newline='\n') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(all_metrics)
                    for k, v in sorted(ordered_metrics.items()):
                        writer.writerow(v)

    def clean(self):        
        self.open_orders = []
        self.trades = []
        self.analytics = None
        self.predictions = {}

    def export_settings(self, settings: Dict):
        s = copy.deepcopy(settings)
        s['code_hash'] = os.popen('git rev-parse HEAD').read().rstrip()
        s['timestamp_started'] = self.started_at.isoformat()
        s['timestamp_finished'] = datetime.now().isoformat()

        if not os.path.isdir(self.experiment_folder):
            os.mkdir(self.experiment_folder)

        timestamp_folder = os.path.join(self.experiment_folder, 'results__' + self.timestamp)
        if not os.path.isdir(timestamp_folder):
            os.mkdir(timestamp_folder)

        with open(os.path.join(timestamp_folder, 'settings.json'), 'w') as fp:
            json.dump(s, fp, indent=4, sort_keys=True)