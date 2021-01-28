from pandas import Timestamp
import uuid

class TradeOrder(object):

    def __init__(self, instrument: int, started: Timestamp):
        self.instrument: int = instrument
        self.started: Timestamp = started
        self.id: str = str(uuid.uuid4())
