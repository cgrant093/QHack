#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    
    def sub_oracle(oracle_num):
        # apply function
        fs[oracle_num]([0, 1, 2])
        
        # return to Z basis for the 3 qubits
        for i in range(3):
            qml.Hadamard(wires=i)
       
        # apply the two CCX gates to one of the auxiliary qubits
            # the aux qb should save whether it's constant (1) or balanced (0)
        qml.Toffoli(wires=[0, 2, oracle_num+3])
        qml.Toffoli(wires=[1, 2, oracle_num+3])
        
        # return back to X basis for the next oracle
        for i in range(3):
            qml.Hadamard(wires=i)
    
    nWires = 7
    dev = qml.device("default.qubit", wires=nWires, shots=1)
    
    @qml.qnode(dev)
    def circuit():
        # state prep
        for i in range(2, nWires):
            qml.PauliX(wires=i)
        
        for i in range(3):
            qml.Hadamard(wires=i)

        # apply each oracle one at a time
        for i in range(len(fs)):
            sub_oracle(i)

        # sample the aux qb since they have the function type saved.
        return qml.sample(wires=[3, 4, 5, 6])

    sample = circuit()

    if np.sum(sample) == len(fs)/2:
        return "2 and 2"
    
    elif np.sum(sample) == (len(fs) or 0):
        return "4 same"
    
    return "bad input"
    
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
