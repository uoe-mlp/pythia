import numpy as np
from pandas import Timestamp

from pythia.agent.trader import BuyAndHoldTrader
from pythia.journal import TradeFill


def test_buy_and_hold_trader_decision():
    prediction = np.array([0,0,1,0,0,0])
    conviction = np.array([0,0,0,0,0,0])

    trader = BuyAndHoldTrader.initialise(6, {})

    trades = trader.act(prediction, conviction, Timestamp(2020, 1, 1), np.array([1,1,1,1,1,1]), True)

    assert len(trades) == 6
    assert trades[0].instrument == 0
    assert trades[0].quantity == 1
    assert trades[0].started == Timestamp(2020, 1, 1)
    assert trades[1].instrument == 1
    assert trades[1].percentage == 0.20
    assert trades[1].started == Timestamp(2020, 1, 1)

    trader.update_portfolio([
        TradeFill(
            instrument=0,
            started=Timestamp(2020, 1, 1),
            value=.12,
            quantity=1,
            completed=Timestamp(2020, 1, 1),
            price=.03,
            direction='sell',
            id='aaa'),
        ] + [
            TradeFill(
            instrument=i+1,
            started=Timestamp(2020, 1, 1),
            value=.1,
            quantity=1,
            completed=Timestamp(2020, 1, 1),
            price=1,
            direction='buy',
            id='aaa')
        for i in range(5)])

    trades = trader.act(prediction, conviction, Timestamp(2020, 1, 2), np.array([1,1,1,1,1,1]), True)

    assert len(trades) == 0