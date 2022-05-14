# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:19:01 2022

@author: burak
"""
import pennylane as qml
import torch
import networkx as nx
from pennylane import numpy as np
number_of_qubits = 5 
n_wires = number_of_qubits
G = nx.erdos_renyi_graph(number_of_qubits, 0.4, seed=123, directed=False)

graph = list(G.edges())

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

dev = qml.device("default.qubit", wires=n_wires, shots=1024)
@qml.qnode(dev, interface='torch')
def circuit(gammas, betas , edge=None, n_layers=1):
    
    for wire in range(number_of_qubits):
        
        qml.Hadamard(wires=wire)
        
    for i in range(n_layers):
        
        U_C(gammas[i])
        U_B(betas[i])
    if(edge == None):
        return qml.sample()
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))
    
n_layers = 2

gammas = 0.01 * np.random.rand( n_layers, requires_grad=True)
betas = 0.01 * np.random.rand( n_layers, requires_grad=True)

gammas = torch.Tensor(gammas)
betas = torch.Tensor(betas)
gammas.requires_grad = True
betas.requires_grad = True

def cost(gammas, betas):
    neg_obj = 0
    for edge in graph:
        # objective for the MaxCut problem
        neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
    print(neg_obj)
    return neg_obj


res = (cost(gammas, betas))
res.backward()



opt = torch.optim.Adam([gammas,betas], lr = 0.1)

steps = 200

def closure():
    opt.zero_grad()
    loss = cost(gammas, betas)
    print(loss)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)

# test:
circuit(gammas, betas, n_layers = 2) 