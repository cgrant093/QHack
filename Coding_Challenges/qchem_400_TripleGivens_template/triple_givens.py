import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #
    
    G3 = np.identity(2**NUM_WIRES)
    
    G3[8, 8] = np.cos(gamma / 2)
    G3[8, 56] = -np.sin(gamma / 2)
    G3[56, 8] = np.sin(gamma / 2)
    G3[56, 56] = np.cos(gamma / 2)
    
    return G3
    
    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    
    # prepare initial state |111000>
    qml.PauliX(0)
    qml.PauliX(1)
    qml.PauliX(2)
    
    # apply single and double excitation gates
    qml.SingleExcitation(angles[0], wires=[0, 5])
    qml.DoubleExcitation(angles[1], wires=[0, 1, 4, 5])
    
    # create and apply triple excitation gate
    G3 = triple_excitation_matrix(angles[2])
    qml.QubitUnitary(G3, wires=[0, 1, 2, 3, 4, 5])

    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
