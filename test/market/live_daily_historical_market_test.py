from pythia.market import LiveDailyHistoricalMarket


def test_live_daily_historical_market_setup():
    params = {
        "features": ["random_target.csv", "random_features.csv"],
        "targets": ["random_features.csv"],
        "trading_cost": 0.0001,
        "features": ["SPY", "DIA", "ONEQ", "IWM"],
        "targets": ["SPY", "DIA", "ONEQ", "IWM"],
        "trading_cost": 0.0001,
        "source": "yahoo",
        "start_date": "01-01-2005",
        "end_date": "05-01-2018",
        "feature_keys": ["Volume", "Close", "Prev. Close"],
        "target_keys": "Close",
    }

    market = LiveDailyHistoricalMarket.initialise(params)
    
    assert market.trading_cost == 1e-4
    assert market.X.shape == (3354, 12)
    assert market.Y.shape == (3354, 5)
    assert market.input_size == 12
    assert market.output_size == 5