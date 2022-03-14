#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    
    norm = np.sqrt(alpha**2 + beta**2)
    generally_entangled = np.array([[alpha, -beta], [beta, alpha]]) / norm
    
    qml.QubitUnitary(generally_entangled, wires=0)
    
    qml.CNOT(wires=[0,1])
  
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    
    def v_basis(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    if x == 0:
        qml.QubitUnitary(v_basis(theta_A0), wires=0)
    elif x == 1:
        qml.QubitUnitary(v_basis(theta_A1), wires=0)
    
    if y == 0:
        qml.QubitUnitary(v_basis(theta_B0), wires=1)
    elif y == 1:
        qml.QubitUnitary(v_basis(theta_B1), wires=1)
    
    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    
    prob_of_winning = 0;
    
    for x in range(2):
        for y in range(2):
            probs = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
            
            if (x*y == 0):
                prob_of_winning += (probs[0] + probs[3])
            
            elif (x*y == 1):
                prob_of_winning += (probs[1] + probs[2])
    
    return (prob_of_winning / len(probs))
    
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return (1 - winning_prob(params, alpha, beta))

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    np.random.seed(0)
    
    init_params = np.random.normal(0, np.pi, 4, requires_grad = True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    steps = 100
    
    conv_tol = 1e-6

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #
        params, prev_prob =  opt.step_and_cost(cost, params)
        
        prob = cost(params)
        
        conv = np.abs(prob - prev_prob)
        
        if (conv <= conv_tol):
            break

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")