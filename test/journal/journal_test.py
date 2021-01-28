from pythia.journal import Journal
from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from pandas import Timestamp

def test_init_empty():
    jr = Journal()
    assert len(jr.open_orders) == 0
    assert len(jr.trades) == 0

def test_storing():
    jr = Journal()
    to = TradeOrder(0, Timestamp(2017, 1, 1, 12))
    jr.store_order([to])
    assert jr.open_orders[0].instrument == 0

def test_fill():
    jr = Journal()
    to = [TradeOrder(0, Timestamp(2017, 1, 1, 12)), TradeOrder(1, Timestamp(2017, 1, 1, 12))]
    tf = [TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id),
        TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[1].id)]
    jr.store_order(to)
    jr.store_fill(tf)
    assert len(jr.trades) == 2

def test_fill_mismatch():
    jr = Journal()
    to = [TradeOrder(0, Timestamp(2017, 1, 1, 12)), TradeOrder(1, Timestamp(2017, 1, 1, 12))]
    tf = [TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id),
        TradeFill(0, Timestamp(2017, 1, 1, 12), 1, 1, Timestamp(2017, 1, 1, 12), 1, "up", to[0].id)]
    jr.store_order(to)
    try:
        jr.store_fill(tf)
        assert(False)
    except ValueError as exc:
        assert(True)
