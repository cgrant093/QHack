#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qfunc_adder(m, wires):
    """Quantum function capable of adding m units to a basic state given as input.

    Args:
        - m (int): units to add.
        - wires (list(int)): list of wires in which the function will be executed on.
    """
    
    qml.QFT(wires=wires)

    # QHACK #
    
    m_binary = np.binary_repr(m, width=len(wires))[::-1]
    
    for i in range(len(m_binary)):
        for j in range(i, len(wires)):
            if int(m_binary[i]) == 1:
                qml.U1(2*np.pi/(2**(j - i + 1)), wires=j)   
    
    # QHACK #

    qml.QFT(wires=wires).inv()
    


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    m = int(inputs[0])
    n_wires = int(inputs[1])
    wires = range(n_wires)

    dev = qml.device("default.qubit", wires=wires, shots=1)

    @qml.qnode(dev)
    def test_circuit():
        # Input:  |2^{N-1}>
        qml.PauliX(wires=0)

        qfunc_adder(m, wires)
        return qml.sample()

    output = test_circuit()   
    print(*output, sep=",")
