from qiskit.primitives import StatevectorEstimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from hamilton import hijij
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def create_hamiltonian(his, jijs, c, n):
    
    terms = []

    for i in range(n):
        terms.append(("I"*(n-i-1)+"Z"+"I"*i, his[i]))
        for j in range(i):
            terms.append(("I"*(n-i-1)+"Z"+"I"*(i-j-1)+"Z"+"I"*j, jijs[j, i]))

    terms.append(("I"*n, c))

    print(terms)

    hc = SparsePauliOp.from_list(terms)

    return hc

volume = 8
volumes = np.array([5, 7, 3])
utilities = np.array([10, 90, 10])

his, jijs, c = hijij(volume, volumes, utilities, 10, 3)
reps = 40
hamiltonian = create_hamiltonian(his, jijs, c, 3)
q = QAOAAnsatz(hamiltonian, reps)

nb_params = q.num_parameters
params = np.random.uniform(0, 2 * np.pi, nb_params)

def f(params):
    """
    On suppose qu’on a deux variables globales.
    q est le circuit paramétré renvoyé par QAOAAnsatz
    hamiltonian est l’hamiltonien généré avec la fonction SparsePauliOp.from_list
    """
    pub = [q, [hamiltonian], [params]]
    estimator = StatevectorEstimator()
    result = estimator.run(pubs=[pub]).result()
    cost = result[0].data.evs[0]

    return cost

best_params = minimize(f, params, args=(), method="COBYLA").x

q.assign_parameters(best_params)

q.measure_all()

sampler = Sampler()
job = sampler.run([q], best_params, shots=5000)
result = job.result()

quasi_dists = result.quasi_dists[0].binary_probabilities()

#get value with max probability

solution = max(quasi_dists, key=quasi_dists.get)
solution = solution[::-1] # Reverse measure order for qiskit

volume = sum([volumes[i] for i, x in enumerate(solution) if x == '1'])
utility = sum([utilities[i] for i, x in enumerate(solution) if x == '1'])

print("Solution: {}, Utility: {}, Volume: {}".format(solution, utility, volume))

# Plotting the histogram
states = list(quasi_dists.keys())
probabilities = list(quasi_dists.values())
plt.bar(states, probabilities)
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Histogram of State Probabilities')
plt.xticks(rotation=90)
plt.show()
