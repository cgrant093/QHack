import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    
    # 4 qubits needed for H-H molecule
    nqubits = 4
    dev = qml.device("default.qubit", wires=nqubits)
    
    # initialize the base circuit used
    basis_state = np.array([1, 1, 0, 0])
    def circuit(theta, wires):
        qml.BasisState(basis_state, wires=wires)
        qml.DoubleExcitation(theta, wires=wires)
    
    # defining the cost function (calcuates ground state energy)
    @qml.qnode(dev)
    def cost_func(theta):
        circuit(theta, wires=range(nqubits))
        return qml.expval(H)
    
    # defining final ground state wavefunction
    @qml.qnode(dev)
    def final_wf(theta):
        circuit(theta, wires=range(nqubits))
        return qml.state()
    
    
    # Minimizing the cost function:
    # Optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    # Initialize parameter theta and initial ground state energy
    theta = np.array(0.0, requires_grad = True)
    
    max_iters = 100
    conv_tol = 1e-6
    
    # for loop that breaks when the convergience is less than the tolerance
    for i in range(max_iters):
        theta, prev_energy = opt.step_and_cost(cost_func, theta)
        
        energy = cost_func(theta)
        
        conv = np.abs(energy - prev_energy)
        
        if conv <= conv_tol:
            break
    
    
    # final g.s. energy and wavefunction
    opt_gs_energy = energy
    opt_gs_wavefunc = final_wf(theta)
    
    return [opt_gs_energy, opt_gs_wavefunc]

    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #
    
    gs_wf_outer_prod = beta*np.outer(ground_state, ground_state)
    H_matrix = qml.utils.sparse_hamiltonian(H).real.toarray()
    
    H1_matrix = H_matrix + gs_wf_outer_prod
    
    return qml.Hermitian(H1_matrix, [0, 1, 2, 3])

    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    
    # 4 qubits needed for H-H molecule
    nqubits = 4
    dev2 = qml.device("default.qubit", wires=nqubits)
    
    # initialize the base circuit used
    basis_state = np.array([1, 1, 0, 0])
    def circuit(theta, wires):
        qml.BasisState(basis_state, wires=wires)
        qml.DoubleExcitation(theta[0], wires=wires)
        qml.SingleExcitation(theta[1], wires=[0,2])
    
    # defining the cost function (calcuates ground state energy)
    @qml.qnode(dev2)
    def cost_func(theta):
        circuit(theta, wires=range(nqubits))
        return qml.expval(H1)
    
    
    # Minimizing the cost function:
    # Optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    # Initialize parameters (theta0, theta1) and initial ground state energy
    np.random.seed(0)
    theta = np.random.normal(0, np.pi, 2, requires_grad = True)
    
    max_iters = 100
    conv_tol = 1e-6
    
    # for loop that breaks when the convergience is less than the tolerance
    for i in range(max_iters):
        theta, prev_energy = opt.step_and_cost(cost_func, theta)
        
        energy = cost_func(theta)
        
        conv = np.abs(energy - prev_energy)
        
        if conv <= conv_tol:
            break
    
    
    # final excited energy
    opt_excited_energy = energy
    
    return opt_excited_energy

    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
