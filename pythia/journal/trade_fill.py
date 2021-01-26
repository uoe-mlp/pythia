from pandas import Timestamp


class TradeFill(object):

    def __init__(self, instrument: str, amount: float, started: Timestamp, completed: Timestamp, price: float):
        self.instrument: str = instrument
        self.amount: float = amount
        self.started: Timestamp = started
        self.completed: Timestamp = completed
        self.price: float = price
