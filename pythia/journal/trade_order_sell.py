from pandas import Timestamp

from .trade_order import TradeOrder


class TradeOrderSell(TradeOrder):

    def __init__(self, instrument: int, started: Timestamp, quantity: float):
        super(TradeOrderSell, self).__init__(instrument, started)
        self.quantity: float = quantity