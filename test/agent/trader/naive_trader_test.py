from torch import Tensor
from pandas import Timestamp

from pythia.agent.trader import NaiveTrader
from pythia.journal import TradeFill


def test_naive_trader_decision():
    prediction = Tensor([0,0,1,0,0,0])
    conviction = Tensor([0,0,0,0,0,0])

    trader = NaiveTrader.initialise(6, {})

    trades = trader.act(prediction, conviction, Timestamp(2020, 1, 1), Tensor([1,1,1,1,1,1]), True)

    assert len(trades) == 2
    assert trades[0].instrument == 0
    assert trades[0].quantity == 1
    assert trades[0].started == Timestamp(2020, 1, 1)
    assert trades[1].instrument == 2
    assert trades[1].percentage == 1
    assert trades[1].started == Timestamp(2020, 1, 1)

def test_naive_trader_update():
    prediction = Tensor([0,0,1,0,0,0])
    conviction = Tensor([0,0,0,0,0,0])

    trader = NaiveTrader.initialise(6, {})

    trader.update_portfolio([
        TradeFill(
            instrument=2,
            started=Timestamp(2020, 1, 1),
            value=.12,
            quantity=4,
            completed=Timestamp(2020, 1, 1),
            price=.03,
            direction='buy',
            id='aaa'),
        TradeFill(
            instrument=4,
            started=Timestamp(2020, 1, 1),
            value=.1,
            quantity=1,
            completed=Timestamp(2020, 1, 1),
            price=1,
            direction='buy',
            id='aaa')])

    assert trader.portfolio[0] == 1
    assert trader.portfolio[2] == 4
    assert trader.portfolio[4] == 1