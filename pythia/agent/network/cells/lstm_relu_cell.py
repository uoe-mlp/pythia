from typing import Tuple
from torch import Tensor, mm, sigmoid, relu


def lstm_relu_cell(input: Tensor, hidden: Tuple[Tensor, Tensor], w_ih: Tensor,
              w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    if b_ih and b_hh:
        gates = mm(input, w_ih.t()) + mm(hx, w_hh.t()) + b_ih + b_hh
    else:
        gates = mm(input, w_ih.t()) + mm(hx, w_hh.t())

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = sigmoid(ingate)
    forgetgate = sigmoid(forgetgate)
    cellgate = relu(cellgate)
    outgate =  sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * relu(cy)

    return hy, cy