import pytest
from torch import Tensor, equal
import numpy as np

from pythia.agent.predictor import FlatPredictor


def test_fit_and_predict():
    lp = FlatPredictor.initialise(1, 1, {})

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [0.0]])
    y_train = np.array([[0.0], [0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])
    lp.fit(X_train, y_train)

    assert lp.predict(X_train[-2:-1,:])[0] == pytest.approx(0, abs=1)
