from pandas import Timestamp
from torch import Tensor
import pytest

from pythia.journal import Analytics
from pythia.journal import TradeFill


def test_analytics():
    timestamps = [
        Timestamp(2021, 1, 1),
        Timestamp(2021, 1, 2),
        Timestamp(2021, 1, 4),
        Timestamp(2021, 1, 5),
        Timestamp(2021, 1, 10),
    ]

    fills = [
        TradeFill(instrument=0, started=timestamps[0], value=0.99, quantity=1, completed=timestamps[0], price=0.99, direction='sell', id='dummy1'),
        TradeFill(instrument=1, started=timestamps[0], value=0.99, quantity=2, completed=timestamps[0], price=0.49, direction='buy', id='dummy2'),
        TradeFill(instrument=1, started=timestamps[0], value=9.90, quantity=2, completed=timestamps[1], price=4.90, direction='sell', id='dummy3'),
        TradeFill(instrument=2, started=timestamps[0], value=9.90, quantity=10, completed=timestamps[1], price=0.99, direction='buy', id='dummy4'),
        TradeFill(instrument=2, started=timestamps[4], value=1, quantity=10, completed=timestamps[4], price=0.1, direction='sell', id='dummy9'),
        TradeFill(instrument=1, started=timestamps[4], value=0.4, quantity=2, completed=timestamps[4], price=0.20, direction='buy', id='dummy10')
    ]

    prices = Tensor([
        [1, 0.5, 3],
        [1, 4.95, 1],
        [1, 10, 0.87],
        [1, 30, 1.3],
        [1, 10, 3],
    ])

    analytics = Analytics.initialise(timestamps=timestamps,fills=fills,prices=prices)

    assert analytics.maximum_drawdown == pytest.approx(-0.13, abs=0.001)
    assert analytics.volatility == pytest.approx(4.2621, abs=0.001)
    assert analytics.cumulative_return == pytest.approx(30, abs=0.001)
    assert analytics.sharpe_ratio == pytest.approx(0.6260, abs=0.001)
    assert analytics.sortino_ratio == pytest.approx(35.5468, abs=0.001)