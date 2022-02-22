#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    
    # cnot.wires returns wires used in CNOT object
        # cnot.wires[0] returns control qubit
        # cnot.wires[1] returns target qubit
    
    # recursively iterates to find all paths from control qubit to target qubit
        # adds 2 swaps per iteration since we'd want to swap back to original qubit order
        # graph_val is the previous value for the graph dictionary in this iteration
        # curr_graph_key is the current key list for the graph dictionary in this iteration    
    
    def one_swap(graph_val, graph_key, num_swaps):
        temp_list = np.empty(0)
        
        # base case if it gets stuck in an endless loop (i.e. 4 -> 7 -> 6 -> 5 -> 4)
        if num_swaps > 16:
            return temp_list
        
        # base case if target is in current dictionary key
        if cnot.wires[1] in graph_key:
            return np.append(temp_list, num_swaps)
   
        for i in range(len(graph_key)):
            next_graph = graph[graph_key[i]]
            
            # if the next dictionary key is only one value
                # and that value is the current dictionary value
                # then ignore this iteration
            if not((len(next_graph) == 1) and (graph_val in next_graph)):
                new_val = one_swap(i, next_graph, num_swaps + 2)
                temp_list = np.append(temp_list, new_val)
        
        return temp_list
    
    # creates list of possible number of swaps needed to make connection
        # between control and target qubit
    nSwap_list = one_swap(cnot.wires[0], graph[cnot.wires[0]], 0)
    
    # returns minimum value of that list (as an integer)
    return int(np.amin(nSwap_list))
    
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
