import os
from pandas import Timestamp

from pythia.journal import Journal
from pythia.journal import TradeOrder
from pythia.journal import TradeFill


def test_init_empty():
    jr = Journal(experiment_folder=os.path.join('test', '.tmp'))
    assert len(jr.open_orders) == 0
    assert len(jr.trades) == 0

def test_storing():
    jr = Journal(experiment_folder=os.path.join('test', '.tmp'))
    to = TradeOrder(0, Timestamp(2017, 1, 1, 12))
    jr.store_order([to])
    assert jr.open_orders[0].instrument == 0

def test_fill():
    jr = Journal(experiment_folder=os.path.join('test', '.tmp'))
    to = [TradeOrder(0, Timestamp(2017, 1, 1, 12)), TradeOrder(1, Timestamp(2017, 1, 1, 12))]
    tf = [TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id),
        TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[1].id)]
    jr.store_order(to)
    jr.store_fill(tf)
    assert len(jr.trades) == 2

def test_fill_mismatch():
    jr = Journal(experiment_folder=os.path.join('test', '.tmp'))
    to = [TradeOrder(0, Timestamp(2017, 1, 1, 12)), TradeOrder(1, Timestamp(2017, 1, 1, 12))]
    tf = [TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id),
        TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id)]
    jr.store_order(to)
    try:
        jr.store_fill(tf)
        assert(False)
    except ValueError as exc:
        assert(True)
