# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:43:52 2022

@author: burak
"""

N_QUBITS = 3


import pennylane as qml
import numpy as np

z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)



def KroneckerProduct(listOfQubits, pauli):
    out = np.array([1])
    for i in range(N_QUBITS):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out


listZ = [[], [0,1], [1,2], [0,2]]

H = sum([KroneckerProduct(z, pauli_Z) for z in listZ])
ew , ev = np.linalg.eig(-1 * H)
ground_state = ev[:, np.argmin(ew)]
lowest_energy = np.argwhere(ew == np.amin(ew))











N_QUBITS = 2
HH = KroneckerProduct([0,1], z) + KroneckerProduct([0,1], x) 
findGroundStates(H)

def findGroundStates(H):
    ew , ev = np.linalg.eig(-1 * H)
    lowest_energy_states = np.argwhere(ew == np.amin(ew))
    ground_state = [ev[:, eigenvalue] for eigenvalue in lowest_energy_states]
    return ground_state



findGroundStates(ToricCodeHamiltonian)
