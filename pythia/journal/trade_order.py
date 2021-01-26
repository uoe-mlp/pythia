from pandas import Timestamp


class TradeOrder(object):

    def __init__(self, instrument: str, amount: float, started: Timestamp):
        self.instrument: str = instrument
        self.amount: float = amount
        self.started: Timestamp = started
