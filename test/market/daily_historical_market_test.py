from pythia.journal.trade_order_buy import TradeOrderBuy
from pythia.market import DailyHistoricalMarket
from pythia.journal import TradeOrderSell, TradeOrderBuy


def test_daily_historical_market_setup():
    params = {
        "features": ["random_target.csv", "random_features.csv"],
        "targets": "random_features.csv",
        "trading_cost": 0.0001
    }

    market = DailyHistoricalMarket.initialise(params)
    
    assert market.trading_cost == 1e-4
    assert market.X.shape.numel() ==  2094 * 22
    assert market.Y.shape.numel() ==  2094 * 11
    assert market.input_size == 22
    assert market.output_size == 11

def test_daily_historical_market_one_trade():
    params = {
        "features": ["random_target.csv", "random_features.csv"],
        "targets": "random_features.csv",
        "trading_cost": 0.01
    }

    market = DailyHistoricalMarket.initialise(params)
    
    t = market.timestamps[3]
    trades = [
        TradeOrderSell(instrument=1, started=t, quantity=100),
        TradeOrderBuy(instrument=1, started=t, percentage=1)
    ]

    fills = market.execute(trades, t)

    assert len(fills) == len(trades)
    assert all([fill.id in [trade.id for trade in trades] for fill in fills])
    assert all([trade.id in [fill.id for fill in fills] for trade in trades])

    sell_fill = [fill for fill in fills if fill.id == trades[0].id][0]
    buy_fill = [fill for fill in fills if fill.id == trades[1].id][0]

    assert sell_fill.quantity == 100
    assert buy_fill.quantity == 100 * 0.99 / 1.01

def test_daily_historical_market_multiple_trades():
    params = {
        "features": ["random_target.csv", "random_features.csv"],
        "targets": "random_features.csv",
        "trading_cost": 0.01
    }

    market = DailyHistoricalMarket.initialise(params)
    
    t = market.timestamps[3]
    trades = [
        TradeOrderSell(instrument=1, started=t, quantity=100),
        TradeOrderBuy(instrument=1, started=t, percentage=0.3),
        TradeOrderBuy(instrument=2, started=t, percentage=0.3),
        TradeOrderBuy(instrument=3, started=t, percentage=0.4),
    ]

    fills = market.execute(trades, t)

    assert len(fills) == len(trades)
    assert all([fill.id in [trade.id for trade in trades] for fill in fills])
    assert all([trade.id in [fill.id for fill in fills] for trade in trades])
