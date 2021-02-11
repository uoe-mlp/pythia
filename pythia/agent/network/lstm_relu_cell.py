from typing import Optional, Tuple
from torch import Tensor
from torch.nn.modules import Module
from torch.nn import Dropout
from torch.nn.modules.rnn import RNNCellBase
import torch

from pythia.agent.network.cells.lstm_relu_cell import lstm_relu_cell


class _LSTMReLUBasicCell(RNNCellBase):

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(_LSTMReLUBasicCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return lstm_relu_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )



class LSTMReLUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, dropout: Optional[float] = None) -> None:
        self.lstm = _LSTMReLUBasicCell(input_size, hidden_size, bias),
        self.input_dropout = Dropout()
        self.recurrent_dropout = Dropout()

    def forward()

