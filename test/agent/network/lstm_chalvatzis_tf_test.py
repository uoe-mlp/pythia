import pandas as pd
import os
import numpy as np
import pytest

from pythia.agent.network import LSTMChalvatzisTF, OutputObserver


def test_lstm_chalvatzis_tf_smoke():
    df = pd.read_csv(os.path.join('test','agent','network','data','chalvatzis_tf.csv'))

    X = df[['x']].values
    Y = df[['y']].values

    window_size = 20

    X_train_ls = []
    Y_train_ls = []
    X_val_ls = []
    Y_val_ls = []
    Y_train_ls = []
    X_test_ls = []
    Y_test_ls = []

    for i in range(X.shape[0] - window_size):
        if i < 0.8 * (X.shape[0] - window_size):
            X_train_ls.append(X[i:i + window_size,:])
            Y_train_ls.append(Y[i:i + window_size,:])
        elif i < 0.9 * (X.shape[0] - window_size):
            X_val_ls.append(X[i:i + window_size,:])
            Y_val_ls.append(Y[i:i + window_size,:])
        else:
            X_test_ls.append(X[i:i + window_size,:])
            Y_test_ls.append(Y[i:i + window_size,:])
    
    X_train = np.array(X_train_ls)
    Y_train = np.array(Y_train_ls)
    X_val = np.array(X_val_ls)
    Y_val = np.array(Y_val_ls)
    X_test = np.array(X_test_ls)
    Y_test = np.array(Y_test_ls)

    net = LSTMChalvatzisTF(input_size=1, window_size=window_size, hidden_size=[16, 16], output_size=1, dropout=[0,0], masked=False)
    net.compile(optimizer='adam', loss='mse', metrics=['mae'])

    net.fit(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val), callbacks=[OutputObserver(net, X_train, Y_hat=Y_train, epochs=5, batch_size=1)])

    net.summary()

    net.evaluate(X_val, Y_val, verbose=1)

    Y_predict = net.predict(X_test)

    assert np.mean(np.abs(Y_predict - Y_test)) < 0.2
