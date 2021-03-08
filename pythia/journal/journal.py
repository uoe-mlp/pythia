from typing import List, Optional, Tuple, Dict
import os
from pandas._libs.tslibs import Timestamp
import numpy as np
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

    def run_analytics(self, type: str, timestamps: List[Timestamp], prices: np.ndarray, instruments: List[str], name: Optional[str]=None, **kwargs):
        self.analytics = Analytics.initialise(timestamps, [x[1] for x in self.trades], prices, self.predictions)
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

    def compile_results(self, epochs_between_validation: int, final_epochs: int):
        timestamp_folder = os.path.join(self.experiment_folder, 'results__' + self.timestamp)

        all_metrics: List = ["Epochs", "Cumulative Return", "Maximum Drawdown", "Mean Directional Accuracy",
                            "Number of Trades", "Sharpe Ratio", "Sortino Ratio", "Volatility"]
        
        # Since the files are processed out of order keep a dictionary
        ordered_metrics: Dict = {}
        final_epochs = final_epochs if final_epochs != 0 else epochs_between_validation

        # Get list of all files
        folders = [f for f in os.listdir(timestamp_folder) if re.match('^train_', f)]
        last_folder_index = len(folders) - 1
        for f in folders:
            subdir = os.path.join(timestamp_folder, f, 'data.json')
            with open(subdir, 'r') as fp:
                data: Dict = json.load(fp)

            index = int(f[6:])
            epochs = (index + 1) * epochs_between_validation if index != last_folder_index else index * epochs_between_validation + final_epochs
            cumulative_return = ArgsParser.get_or_error(data, 'cumulative_return')
            maximum_drawdown = ArgsParser.get_or_error(data, 'maximum_drawdown')
            mean_directional_accuracy = ArgsParser.get_or_error(data, 'mean_directional_accuracy')
            number_of_trades = ArgsParser.get_or_error(data, 'number_of_trades')
            sharpe_ratio = ArgsParser.get_or_error(data, 'sharpe_ratio')
            sortino_ratio = ArgsParser.get_or_error(data, 'sortino_ratio')
            volatility = ArgsParser.get_or_error(data, 'volatility')

            ordered_metrics[index] = [epochs, cumulative_return, maximum_drawdown, mean_directional_accuracy, number_of_trades,
                                sharpe_ratio, sortino_ratio, volatility]

        sorted_dict = dict(sorted(ordered_metrics.items()))

        output_csv = os.path.join(timestamp_folder, "train_full")
        if not os.path.isdir(output_csv):
            os.mkdir(output_csv)
        output_csv = os.path.join(output_csv, "output.csv")
        # Write to csv
        with open(output_csv, 'w+', newline='\n') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(all_metrics)
                for k, v in sorted_dict.items():
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