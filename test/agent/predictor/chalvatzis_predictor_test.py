import numpy as np
import pandas as pd
import os

from pythia.agent.predictor import ChalvatzisPredictor


def test_initialise():
    cp = ChalvatzisPredictor.initialise(input_size=2, output_size=3,
                                        params=dict(window_size=12, hidden_size=64, epochs=100, predict_returns=False,
                                                    shuffle=False, dropout=0.5, all_hidden=False))
    cp.model.build(input_shape=(None, 12, 2,))
    cp.model.summary()
    assert True


def test_sequence_creation_two_dim():
    cp = ChalvatzisPredictor.initialise(input_size=2, output_size=2, params=dict(window_size=3, hidden_size=64,
                                                                                 epochs=100, predict_returns=False, shuffle=False, dropout=0.5, all_hidden=False))

    X_train = np.array([[0.0, 1.5], [1.0, 1.5], [2.0, 1.5], [3.0, 1.5], [4.0, 1], [
                       5.0, 1], [6.0, 1], [7.0, 1], [8.0, 1], [9.0, 1], [10.0, 1], [0.0, 1.5]])
    Y_train = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [4.0, 0.1], [6.0, 0.1], [8.0, 0.1], [
                       10.0, 0.1], [12.0, 0.1], [14.0, 0.1], [16.0, 0.1], [18.0, 0.1], [20.0, 0.1]])
    data = cp._ChalvatzisPredictor__create_sequences(X_train, Y_train, [7, 11])

    assert data[0][0].shape == (5, 3, 2)
    assert data[0][1].shape == (5, 3, 2)
    assert data[1][0].shape == (4, 3, 2)
    assert data[1][1].shape == (4, 3, 2)
    assert data[2][0].shape == (1, 3, 2)
    assert data[2][1].shape == (1, 3, 2)

    np.testing.assert_almost_equal(
        data[0][1][3, :, :], np.array([[4., 0.1], [6., 0.1], [8., 0.1]]))
    np.testing.assert_almost_equal(
        data[1][0][0, :, :], np.array([[5., 1.], [6., 1.], [7., 1.]]))
    np.testing.assert_almost_equal(
        data[2][0][-1, :, :], np.array([[9., 1.], [10., 1.], [0, 1.5]]))


def test_fit_and_predict():
    np.random.seed(12345)
    cp = ChalvatzisPredictor.initialise(input_size=1, output_size=1,
                                        params=dict(window_size=3, hidden_size=64, epochs=1, predict_returns=False, shuffle=False, dropout=0.5, all_hidden=False))

    df = pd.read_csv(os.path.join('test','agent','network','data','chalvatzis_tf.csv'))

    X = df[['x']].values
    Y = df[['y']].values

    cp.fit(X[:-20,:], Y[:-20,:], X[-20:-10,:], Y[-20:-10,:])

    Y_test = cp.predict(X[:-4,:])

    assert Y_test[0].shape == (1,)


def test_normalization():
    np.random.seed(12345)
    cp = ChalvatzisPredictor.initialise(input_size=2, output_size=1,
                                        params=dict(window_size=3, hidden_size=64, epochs=1, predict_returns=False, shuffle=False, dropout=0.5, all_hidden=False, normalize={'active': True, 'min': -3}))

    assert cp.normalize_min == -3
    assert cp.normalize_max == 1

    X_train = np.array([[0.0, 1.5], [1.0, 1.5], [2.0, 1.5], [3.0, 1.5], [4.0, 1], [5.0, 1], [6.0, 1], [7.0, 1], [8.0, 1], [9.0, 1], [10.0, 1], [0.0, 1.5]])    
    y_train = np.array([[0.0], [0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])

    cp._ChalvatzisPredictor__normalize_fit(X_train, y_train)

    np.testing.assert_equal(cp._normalize_fitted_min_feature, np.array([0., 1.]))
    np.testing.assert_equal(cp._normalize_fitted_max_feature, np.array([10., 1.5]))

    X_normalized = cp._ChalvatzisPredictor__normalize_apply_features(X_train)

    assert X_normalized[0,0] == -3
    assert X_normalized[0,1] == 1
    assert X_normalized[4,1] == -3
    assert X_normalized[10,0] == 1
    assert X_normalized[5,0] == -1

    cp.fit(X_train, y_train)
    cp.predict(X_train[-10:,:])

def test_fit_and_predict_shuffle_smoke():
    np.random.seed(12345)
    cp = ChalvatzisPredictor.initialise(input_size=1, output_size=1,
                                        params=dict(window_size=3, shuffle=True, hidden_size=64, epochs=1, predict_returns=False, dropout=0.5, all_hidden=False))

    df = pd.read_csv(os.path.join('test','agent','network','data','chalvatzis_tf.csv'))

    X = df[['x']].values
    Y = df[['y']].values

    cp.fit(X[:-20,:], Y[:-20,:], X[-20:-10,:], Y[-20:-10,:])

    Y_test = cp.predict(X[:-4,:])

    assert Y_test[0].shape == (1,)

def test_fit_and_predict_multiple_iter():
    np.random.seed(12345)
    cp = ChalvatzisPredictor.initialise(input_size=3, output_size=2,
                                        params=dict(window_size=5, shuffle=True, iter_per_item=2, hidden_size=64, epochs=1, predict_returns=False, dropout=0.5, all_hidden=False))

    df = pd.read_csv(os.path.join('test','agent','network','data','chalvatzis_tf.csv'))

    X = np.concatenate([df[['x']].values, df[['x']].values + 1, df[['x']].values - 1], axis=1)
    Y =  np.concatenate([df[['y']].values, df[['y']].values - 1], axis=1)

    cp.fit(X[:-20,:], Y[:-20,:], X[-20:-10,:], Y[-20:-10,:])

    predictions, convictions = cp.predict(X, all_history=True)
    assert predictions.shape == (X.shape[0] - cp.window_size + 1, Y.shape[1])