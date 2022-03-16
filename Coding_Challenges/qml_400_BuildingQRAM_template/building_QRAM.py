#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.
    
    def A_matrix():
        ket = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
        
        A = np.zeros((16, 16))
        
        def matrix_RY(theta):
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            return np.array([[c, -s], [s, c]])
        
        for i in range(8):
            m = np.asarray(list(np.binary_repr(i, width=3))).astype(int)
            
            ket_ij = np.kron(ket[m[0]], ket[m[1]])
            ket_kA = np.kron(ket[m[2]], matrix_RY(thetas[i]))
            
            A += np.kron(ket_ij, ket_kA)
        
        return A

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
    
        for i in range(3):
            qml.Hadamard(i)
        
        qml.QubitUnitary(A_matrix(), wires=[0, 1, 2, 3])
        
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
