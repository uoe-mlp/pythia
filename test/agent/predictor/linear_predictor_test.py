import pytest
from torch import Tensor

from pythia.agent.predictor import LinearPredictor


def test_check_base_params():
    lp = LinearPredictor.initialise(1, 1, {})

    assert lp.weight_decay == 0.0
    assert lp.learning_rate == 1e-3
    assert lp.epochs == 100

def test_check_weights():
    lp = LinearPredictor.initialise(1, 1, {"learning_rate": 0.01, "epochs": 2500})

    X_train = Tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    y_train = Tensor([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])
    lp.fit(X_train, y_train)

    assert lp.model.linear.weight.item() == pytest.approx(2.0, abs=0.1, rel=0.1)

def test_check_biases():
    lp = LinearPredictor.initialise(1, 1, {"learning_rate": 0.01, "epochs": 3000})

    X_train = Tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    y_train = Tensor([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])
    lp.fit(X_train, y_train)

    assert lp.model.linear.bias.item() == pytest.approx(0.0, abs=0.2, rel=0.2)

def test_smoke_two_dim():
    lp = LinearPredictor.initialise(2, 2, {"learning_rate": 0.01, "epochs": 3000})

    X_train = Tensor([[0.0, 1.5], [1.0, 1.5], [2.0, 1.5], [3.0, 1.5], [4.0, 1], [5.0, 1], [6.0, 1], [7.0, 1], [8.0, 1], [9.0, 1], [10.0, 1]])
    Y_train = Tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.1], [6.0, 0.1], [8.0, 0.1], [10.0, 0.1], [12.0, 0.1], [14.0, 0.1], [16.0, 0.1], [18.0, 0.1], [20.0, 0.1]])
    lp.fit(X_train, Y_train)

    assert True