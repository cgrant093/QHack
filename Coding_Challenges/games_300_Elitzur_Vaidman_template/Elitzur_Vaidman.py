#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1, shots=1)


@qml.qnode(dev)
def is_bomb(angle):
    """Construct a circuit at implements a one shot measurement at the bomb.

    Args:
        - angle (float): transmissivity of the Beam splitter, corresponding
        to a rotation around the Y axis.

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    
    qml.RY(2*angle, wires=0)

    # QHACK #

    return qml.sample(qml.PauliZ(0))


@qml.qnode(dev)
def bomb_tester(angle):
    """Construct a circuit that implements a final one-shot measurement, given that the bomb does not explode

    Args:
        - angle (float): transmissivity of the Beam splitter right before the final detectors

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #

    qml.RY(2*angle, wires=0)
    qml.PauliX(0)

    # QHACK #

    return qml.sample(qml.PauliZ(0))


def simulate(angle, n):
    """Concatenate n bomb circuits and a final measurement, and return the results of 10000 one-shot measurements

    Args:
        - angle (float): transmissivity of all the beam splitters, taken to be identical.
        - n (int): number of bomb circuits concatenated

    Returns:
        - (float): number of bombs successfully tested / number of bombs that didn't explode.
    """

    # QHACK #
    
    n_shots = 10000
    C_beeps = 0
    D_beeps = 0
    explosions = 0
    
    def n_bomb_circ(i):
    
        if (i == n):
            final_test = bomb_tester(angle)
          
            if (final_test == 1):
                return [0, 1, 0]
            
            return [0, 0, 1]
        
        bomb_pot = is_bomb(angle)
        
        if (bomb_pot == 1):
            return [1, 0, 0]
        
        return n_bomb_circ(i+1)
    

    for j in range(n_shots):
        results = n_bomb_circ(0)

        explosions += results[0]
        D_beeps += results[1]
        C_beeps += results[2]
    
    print(D_beeps)
    print(n_shots - explosions)
    
    return D_beeps / (n_shots - explosions)

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = simulate(float(inputs[0]), int(inputs[1]))
    print(f"{output}")
