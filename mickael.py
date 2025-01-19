from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from math import floor, pi
import numpy as np

class Knapsack:
    """
    Class to represent a knapsack problem.
    """
    def __init__(self, max_volume, volume, utilities):
        self.max_volume = max_volume
        self.volume = volume
        self.utilities = utilities

def solve_knapsack(knapsack):
    """
    Function to solve the knapsack problem using dynamic programming.
    """
    num_items = len(knapsack.volume)
    dp_table = [[0 for _ in range(knapsack.max_volume + 1)] for _ in range(num_items + 1)]
    for item_index in range(1, num_items + 1):
        for current_volume in range(1, knapsack.max_volume + 1):
            if knapsack.volume[item_index - 1] <= current_volume:
                dp_table[item_index][current_volume] = max(
                    dp_table[item_index - 1][current_volume],
                    dp_table[item_index - 1][current_volume - knapsack.volume[item_index - 1]] + knapsack.utilities[item_index - 1]
                )
            else:
                dp_table[item_index][current_volume] = dp_table[item_index - 1][current_volume]

    if dp_table[num_items][knapsack.max_volume] == 0:
        return 0, []

    selected_items = []
    remaining_volume = knapsack.max_volume
    for item_index in range(num_items, 0, -1):
        if dp_table[item_index][remaining_volume] != dp_table[item_index - 1][remaining_volume]:
            selected_items.append(item_index - 1)
            remaining_volume -= knapsack.volume[item_index - 1]

    if remaining_volume != 0:
        return 0, []

    selected_items.reverse()
    return dp_table[num_items][knapsack.max_volume], selected_items, sum(knapsack.volume[i] for i in selected_items)

def get_possible_combinations_with_utility_over_k(knapsack, k):
    """
    Function to get all possible combinations of items that have a utility greater than k.
    """
    possible_combinations = []
    utilities = []
    n = len(knapsack.utilities)

    for subset in range(1 << n):
        selected_utility = sum(knapsack.utilities[i] for i in range(n) if (subset & (1 << i)))
        selected_volume = sum(knapsack.volume[i] for i in range(n) if (subset & (1 << i)))

        if selected_volume <= knapsack.max_volume and selected_utility >= k:
            possible_combinations.append(subset)
            utilities.append(selected_utility)

    return possible_combinations, utilities

def mark_solutions_with_oracle(circuit, qubits, L):
    """
    Function to mark the solutions in L with an oracle for Grover's algorithm.
    """
    n = len(qubits)
    for solution in L:
        binary_solution = format(solution, f'0{n}b')
        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qubits[i])

        circuit.h(qubits[-1])
        circuit.mcx(qubits[:-1], qubits[-1])
        circuit.h(qubits[-1])

        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qubits[i])

def q13(knapsack, k):
    """
    Function to solve question 13: Constructs a list L of all realizable solutions with weight > k,
    and builds an oracle for Grover's algorithm.
    """
    solutions, _ = get_possible_combinations_with_utility_over_k(knapsack, k)

    n = len(knapsack.utilities)
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))

    mark_solutions_with_oracle(qc, list(range(n)) + [n], solutions)

    return qc

def apply_diffusion(circuit, qubits):
    """
    Function to apply the diffusion operator to the given circuit.
    """
    circuit.h(qubits)
    circuit.x(qubits)
    circuit.h(qubits[-1])
    circuit.mcx(qubits[:-1], qubits[-1])
    circuit.h(qubits[-1])
    circuit.x(qubits)
    circuit.h(qubits)

def solve_knapsack_optimization(knapsack):
    """
    Function to solve the knapsack problem using Grover's algorithm.
    """
    solutions, utilities = get_possible_combinations_with_utility_over_k(knapsack, 1)
    max_utility = max(utilities)
    optimal_solutions = [solutions[i] for i, u in enumerate(utilities) if u == max_utility]

    n = len(knapsack.utilities)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    num_grover_iterations = floor(pi * np.sqrt(2**n / len(optimal_solutions)) / 4)
    for _ in range(num_grover_iterations):
        mark_solutions_with_oracle(qc, list(range(n)), optimal_solutions)
        apply_diffusion(qc, list(range(n)))

    qc.measure(range(n), range(n))

    sampler = Sampler()
    job = sampler.run([qc])
    result = job.result()

    quasi_dists = result.quasi_dists[0].binary_probabilities()
    probabilities = {state: prob for state, prob in quasi_dists.items()}

    max_state = max(probabilities, key=probabilities.get)
    selected_utility = sum(knapsack.utilities[i] for i in range(n) if max_state[i] == '1')

    return max_state, selected_utility

if __name__ == '__main__':
    knapsack = Knapsack(2, [2, 1, 1], [1, 2, 1])

    utility, items, volume = solve_knapsack(knapsack) 
    print("Classic Knapsack Solution:")
    print("Utility:", utility)
    print("Items Included:", items)
    print("Total Volume:", volume)

    print("\nQuantum Knapsack with Grover :")
    result_state, result_utility = solve_knapsack_optimization(knapsack)
    print(f"Selected State: {result_state}, Total Utility: {result_utility}")
