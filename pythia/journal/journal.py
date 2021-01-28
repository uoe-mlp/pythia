from typing import List, Optional, Tuple

from .trade_order import TradeOrder
from .trade_fill import TradeFill


class Journal(object):

    def __init__(self):
        self.open_orders: List[TradeOrder] = []
        self.trades: List[Tuple[TradeOrder, TradeFill]] = []

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
