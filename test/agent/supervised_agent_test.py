import pytest
import numpy as np
from pythia.journal import TradeFill
from pythia.agent.predictor import FlatPredictor
from pythia.agent.trader import BuyAndHoldTrader

from pandas import Timestamp

from pythia.agent import SupervisedAgent


def test_check_initialisation():
    agent = SupervisedAgent.initialise(1, 1, {})
    assert isinstance(agent.predictor, FlatPredictor)
    assert isinstance(agent.trader, BuyAndHoldTrader)

def test_fit():
    sa = SupervisedAgent.initialise(1, 1, {"predictor": {"type": "linear", "params": {"learning_rate": 0.01, "epochs": 2500}}})

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])
    sa.fit(X_train, y_train, X_train, y_train, [])

    assert sa.predictor.model.linear.weight.item() == pytest.approx(2.0, abs=0.1, rel=0.1)


def test_prediction_act():
    sa = SupervisedAgent.initialise(1, 2, {"predictor": {"type": "linear", "params": {"learning_rate": 0.01, "epochs": 2500}}})

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    Y_train = np.array([[0.0, 0.0], [0.0,2.0], [0.0,4.0], [0.0,6.0], [0.0,8.0], [0.0,10.0], [0.0,12.0], [0.0,14.0], [0.0,16.0], [0.0,18.0], [0.0,20.0]])
    X_test = np.array([[11.0]])
    Y_test = np.array([[1,10]])

    sa.fit(X_train, Y_train, X_train, Y_train, [])

    assert sa.act(X_test, Timestamp(2017, 1, 1, 12), Y_test)[0][0].instrument == 0

def test_portfolio_update():
    agent = SupervisedAgent.initialise(1, 1, {})
    pf = TradeFill(0, Timestamp(2017, 1, 1, 12), 0, 0, Timestamp(2017, 1, 1, 12), 1, "buy", "uuid")
    agent.update([pf], np.array([[0]]), np.array([[1], [1]]))

    assert agent.trader._portfolio[0] == 1