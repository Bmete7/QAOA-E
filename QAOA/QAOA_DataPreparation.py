# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:10:31 2022

@author: burak
"""

from pennylane import numpy as np
import qiskit
import time
import networkx as nx
import pennylane as qml
import timeit

# %% Problem definition
number_of_qubits = 5
G = nx.Graph()


for i in range(5):
    G.add_node(i)

G.add_edge(0,1)

G.add_edge(0,2)
G.add_edge(0,3)
G.add_edge(0,4)

G.add_edge(1,5)
G.add_edge(1,6)
G.add_edge(1,7)

G.add_edge(2,5)
G.add_edge(3,6)
G.add_edge(4,7)
G = nx.erdos_renyi_graph(number_of_qubits, 0.4, seed=123, directed=False)


nx.draw(G, with_labels=True, alpha=0.8)

graph = list(G.edges)


createBitString = lambda x: str(bin(x)[2:].zfill(number_of_qubits))
decomposeBitString = lambda b: int("".join(str(bs) for bs in b), base = 2)

int2bit = lambda x: str(bin(x)[2:].zfill(number_of_qubits))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)

dev = qml.device("default.qubit", wires=number_of_qubits, shots=1024)


# np.sort(a)

# createBitString(15)

def edgeCount(solution, G):
  edge_count = 0
  edges = G.edges()
  for edge in edges:
    edge_1, edge_2 = edge
    if(solution[edge_1] != solution[edge_2]):
      edge_count += 1
  return edge_count * -1

solution_dictionary = {}
solution_array = []
for i in range(2**5):
    solution_dictionary[createBitString(i)] = edgeCount(createBitString(i), G)
    solution_array.append(edgeCount(createBitString(i), G))

np.argmin(solution_array)
np.min(solution_array)

int2bit(np.argmin(solution_array))


#%%
reduced_dict = {}
reduced_array = np.zeros_like(solution_array)
for item in solution_dictionary.items():
    keys,val = item
    if keys.startswith('00'):
        reduced_dict[keys] = val
        reduced_array[bit2int(keys)] = val
        
np.argmin(reduced_array)
np.min(reduced_array)


# %% 
# unitary operator U_B with parameter beta
n_wires = 5

def U_B(beta):
    for wire in range(n_wires):
        # if(wire < 2):
        #     continue
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        
        wire1 = edge[0]
        wire2 = edge[1]
        # for i in range(2):
        #     if(wire1 == i or wire2 == i):
        #         continue
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])
        
        
def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)




pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z, requires_grad=False)


@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    
    for wire in range(n_wires):
        # if(wire < 2 ):
        #     continue
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
        print(neg_obj)
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))
        print(bit_strings)

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:06b}".format(most_freq_bit_string))

    return -objective(params), bit_strings


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists
# bitstrings1 = qaoa_maxcut(n_layers=1)[1]
bitstrings2 = qaoa_maxcut(n_layers=2)[1]

import matplotlib.pyplot as plt

xticks = range(0, 17)
xtick_labels = list(map(lambda x: format(x, "05b"), xticks))
bins = np.arange(0, 17) - 0.5

plt.subplot(1, 2, 2)
plt.title("n_layers=2")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.show()
# %% 


# %% 

def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))
    start_time = timeit.time.time()
    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, 3, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.05)

    # optimize parameters in objective
    params = init_params
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(decomposeBitString(circuit(params[0], params[1], edge=None, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))
    
    end_time = timeit.time.time()
    print('Time elapsed {:.2f}'.format(end_time - start_time))
    return -objective(params), bit_strings


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists

bitstrings2 = qaoa_maxcut(n_layers=1)[1]
