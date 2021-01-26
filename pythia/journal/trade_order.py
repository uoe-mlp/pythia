from pandas import Timestamp


class TradeOrder(object):

    def __init__(self, instrument: int, started: Timestamp):
        self.instrument: int = instrument
        self.started: Timestamp = started





