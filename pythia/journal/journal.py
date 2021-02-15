from typing import List, Optional, Tuple
import os
from pandas._libs.tslibs import Timestamp
import numpy as np
from datetime import datetime
import json

from .trade_order import TradeOrder
from .trade_fill import TradeFill
from .analytics import Analytics


class Journal(object):

    def __init__(self, experiment_folder: str):
        self.experiment_folder: str = experiment_folder
        self.open_orders: List[TradeOrder] = []
        self.trades: List[Tuple[TradeOrder, TradeFill]] = []
        self.analytics: Optional[Analytics] = None
        self.timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    def store_order(self, orders: List[TradeOrder]):
        self.open_orders += orders

    def store_fill(self, fills: List[TradeFill]):
        for fill in fills:
            same_id = [x for x in self.open_orders if x.id == fill.id]
            if len(same_id) == 1:
                self.trades.append((same_id[0], fill))
                self.open_orders = [x for x in self.open_orders if x.id != fill.id]
            else:
                raise ValueError('One and only one open order should match the id for this fill.')

    def run_analytics(self, type: str, timestamps: List[Timestamp], prices: np.ndarray):
        self.analytics = Analytics.initialise(timestamps, [x[1] for x in self.trades], prices)

        if not os.path.isdir(self.experiment_folder):
            os.mkdir(self.experiment_folder)

        timestamp_folder = os.path.join(self.experiment_folder, self.timestamp)
        if not os.path.isdir(timestamp_folder):
            os.mkdir(timestamp_folder)

        type_folder = os.path.join(timestamp_folder, type)
        if not os.path.isdir(type_folder):
            os.mkdir(type_folder)

        with open(os.path.join(type_folder, 'data.json'), 'w') as fp:
            json.dump(self.analytics.to_dict(), fp, indent=4, sort_keys=True)

    def clean(self):        
        self.open_orders = []
        self.trades = []
        self.analytics = None
