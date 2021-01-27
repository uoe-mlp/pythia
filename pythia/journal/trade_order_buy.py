from pandas import Timestamp

from .trade_order import TradeOrder


class TradeOrderBuy(TradeOrder):
    
    def __init__(self, instrument: int, started: Timestamp, percentage: float):
        super(TradeOrderBuy, self).__init__(instrument, started)
        self.percentage: float = percentage