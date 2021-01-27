import os
import pytest

from pythia.market import DailyHistoricalMarket


def test_daily_historical_market_setup():
    params = {
        "features": ["random_target.csv", "random_features.csv"],
        "target": "random_features.csv",
        "trading_cost": 0.0001
    }

    market = DailyHistoricalMarket.initialise(params)
    
    assert market.trading_cost == 1e-4
    assert market.X.shape.numel() ==  2094 * 22
    assert market.Y.shape.numel() ==  2094 * 11
    assert market.input_size == 22
    assert market.output_size == 11