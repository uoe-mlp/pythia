from pandas import Timestamp


class TradeFill(object):

    def __init__(self, instrument: int, started: Timestamp, value: float, quantity: float, completed: Timestamp, price: float, direction: str):
        self.instrument: int = instrument
        self.started: Timestamp = started
        self.value: float = value
        self.quantity: float = quantity
        self.completed: Timestamp = completed
        self.price: float = price
        self.direction: str = direction
