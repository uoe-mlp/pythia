from torch import Tensor
from pythia.agent.network import LSTMReLU


def test_lstm_relu():
    net = LSTMReLU(2, 3, 2, False, True, 0.1, False)
    x = Tensor([[[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]]])
    y_est = net.forward(x)
