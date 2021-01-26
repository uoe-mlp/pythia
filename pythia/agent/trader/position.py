from pandas import Timestamp


class Position(object):

    def __init__(self, instrument: str, amount: float):
        self.instrument: str = instrument
        self.amount: float = amount
