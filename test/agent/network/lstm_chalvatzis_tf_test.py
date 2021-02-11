import pandas as pd
import os
import numpy as np
from random import shuffle

from pythia.agent.network import LSTMChalvatzisTF


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
        if i < 0.9 * (X.shape[0] - window_size):
            X_train_ls.append(X[i:i + window_size,:])
            Y_train_ls.append(Y[i:i + window_size,:])
        elif i < 0.95 * (X.shape[0] - window_size):
            X_val_ls.append(X[i:i + window_size,:])
            Y_val_ls.append(Y[i:i + window_size,:])
        else:
            X_test_ls.append(X[i:i + window_size,:])
            Y_test_ls.append(Y[i:i + window_size,:])
    
    train_reindex = list(range(len(X_train_ls)))
    # shuffle(train_reindex)
    X_train_ls = [i for _,i in sorted(zip(train_reindex,X_train_ls))]
    Y_train_ls = [i for _,i in sorted(zip(train_reindex,Y_train_ls))]
    X_train = np.array(X_train_ls)
    Y_train = np.array(Y_train_ls)
    X_val = np.array(X_val_ls)
    Y_val = np.array(Y_val_ls)
    X_test = np.array(X_test_ls)
    Y_test = np.array(Y_test_ls)

    net = LSTMChalvatzisTF(1, window_size, 32, 1, dropout=0.5)

    net.describe()

    net.fit(X_train, Y_train, X_val, Y_val, epochs=2, batch_size=2)

    net.evaluate(X_val, Y_val)

    Y_predict = net.predict(X_test)

    assert True
