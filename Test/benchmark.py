# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:42:18 2022

@author: burak
"""

import pennylane as qml
import networkx as nx
import time
from pennylane import numpy as np
dev_3_mixers_benchmark = qml.device("default.qubit", wires= 9 , shots= 1024)
dev_4_mixers_benchmark = qml.device("default.qubit", wires= 16 , shots= 1024)
import sys
sys.path.append('..\\')
from Utils import *
from Ansatz import *
from Visualization.visualizations import *
import pickle
@qml.qnode(dev_3_mixers_benchmark)
def circuit_3_mixers_benchmark(params, edges):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional
    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    

    n = 3
    for edge in edges:
        u,v = edge
        
        q1 = u * n 
        q2 = v * n 
        for s1 in [True, False]:
            
            for s2 in [True, False]:
                for s3 in [True, False]:
                    
                    if(s1):
                        qml.Hadamard(q1)
                        qml.Hadamard(q2)
                    else:
                        qml.RX(params[0], wires = q1)
                        qml.RX(params[0], wires = q2)
                    if(s2):
                        qml.Hadamard(q1 + 1)
                        qml.Hadamard(q2 + 1)
                    else:
                        qml.RX(params[0], wires = q1 + 1)
                        qml.RX(params[0], wires = q2 + 1)
                    
                    if(s3):
                        qml.Hadamard(q1 + 2)
                        qml.Hadamard(q2 + 2)
                    else:
                        qml.RX(params[0], wires = q1 + 2)
                        qml.RX(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    qml.RZ(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    if(s3):
                        qml.Hadamard(q1 + 2).inv()
                        qml.Hadamard(q2 + 2).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 2).inv()
                        qml.RX(params[0], wires = q2 + 2).inv()    
                    
                    if(s2):
                        qml.Hadamard(q1 + 1).inv()
                        qml.Hadamard(q2 + 1).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 1).inv()
                        qml.RX(params[0], wires = q2 + 1).inv()
                    
                    if(s1):
                        qml.Hadamard(q1).inv()
                        qml.Hadamard(q2).inv()
                    else:
                        qml.RX(params[0], wires = q1).inv()
                        qml.RX(params[0], wires = q2).inv()
                    
    return qml.probs(wires = 0)


@qml.qnode(dev_4_mixers_benchmark)
def circuit_4_mixers_benchmark(params, edges):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional
    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    n = 4

   
    for edge in edges:
        u,v = edge
        
        q1 = u * n
        q2 = v * n
        for s1 in [True, False]:
            for s2 in [True, False]:
                for s3 in [True, False]:
                    for s4 in [True, False]:
                    
                        if(s1):
                            qml.Hadamard(q1)
                            qml.Hadamard(q2)
                        else:
                            qml.RX(params[0], wires = q1)
                            qml.RX(params[0], wires = q2)
                        if(s2):
                            qml.Hadamard(q1 + 1)
                            qml.Hadamard(q2 + 1)
                        else:
                            qml.RX(params[0], wires = q1 + 1)
                            qml.RX(params[0], wires = q2 + 1)
                        
                        if(s3):
                            qml.Hadamard(q1 + 2)
                            qml.Hadamard(q2 + 2)
                        else:
                            qml.RX(params[0], wires = q1 + 2)
                            qml.RX(params[0], wires = q2 + 2)
                        
                        if(s4):
                            qml.Hadamard(q1 + 3)
                            qml.Hadamard(q2 + 3)
                        else:
                            qml.RX(params[0], wires = q1 + 3)
                            qml.RX(params[0], wires = q2 + 3)
                            
                        qml.CNOT(wires= [q1, q1+1])
                        qml.CNOT(wires= [q1+1, q1+2])
                        qml.CNOT(wires= [q1+2, q1+3])
                        qml.CNOT(wires= [q1+3, q2])
                        qml.CNOT(wires= [q2, q2+1])
                        qml.CNOT(wires= [q2+1, q2+2])
                        qml.CNOT(wires= [q2+2, q2+3])
                        
                        qml.RZ(params[0], wires = q2 + 3)
                        
                        qml.CNOT(wires= [q1, q1+1])
                        qml.CNOT(wires= [q1+1, q1+2])
                        qml.CNOT(wires= [q1+2, q1+3])
                        qml.CNOT(wires= [q1+3, q2])
                        qml.CNOT(wires= [q2, q2+1])
                        qml.CNOT(wires= [q2+1, q2+2])
                        qml.CNOT(wires= [q2+2, q2+3])
                        
                        if(s4):
                            qml.Hadamard(q1 + 3).inv()
                            qml.Hadamard(q2 + 3).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 3).inv()
                            qml.RX(params[0], wires = q2 + 3).inv()
                            
                        if(s3):
                            qml.Hadamard(q1 + 2).inv()
                            qml.Hadamard(q2 + 2).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 2).inv()
                            qml.RX(params[0], wires = q2 + 2).inv()    
                        
                        if(s2):
                            qml.Hadamard(q1 + 1).inv()
                            qml.Hadamard(q2 + 1).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 1).inv()
                            qml.RX(params[0], wires = q2 + 1).inv()
                        
                        if(s1):
                            qml.Hadamard(q1).inv()
                            qml.Hadamard(q2).inv()
                        else:
                            qml.RX(params[0], wires = q1).inv()
                            qml.RX(params[0], wires = q2).inv()
                    
    return qml.probs(wires = 0)





def BenchmarkResults():
    with open("data/edges_pickle", "rb") as fp:   # Unpickling
        edges = pickle.load(fp)
    
    adj = np.load('data/adj.npy')
    
    n_layers = 1 
    
    init_params = [1,1]
    
    
    dev = qml.device("default.qubit", wires= 1, shots=1024)
    n_settings = [3,4]
    

    # ansatz_list = [circuit_3_benchmark, circuit_4_benchmark, circuit_3_mixers_benchmark, circuit_4_mixers_benchmark ]
    ansatz_names = ['QAOA-E1-3', 'QAOA-E1-4', 'QAOA-M1-3', 'QAOA-M1-4']
    results = []
    durations = []

      
    @qml.qnode(dev)
    def circuit_benchmark(params, n):
        U_E(n)
        # Discarding the subsystem B, via swapping with a reference system
        discardQubits(n, 0)
            
        return qml.probs(range(0,1))  
      
    for n in n_settings:
        N = n**2
        ancillary = calculateRequiredAncillaries(n)
        dev = qml.device("default.qubit", wires= N + ((N-ancillary) * n_layers), shots=1024)
        start = time.time()
        results.append(qml.specs(circuit_benchmark)(init_params, n))
        end = time.time()
        print('QAOA-E with {} qubits is run in {} seconds'.format(n, end-start))
        durations.append(end-start)

    start = time.time()
    results.append(qml.specs(circuit_3_mixers_benchmark)(init_params, edges))
    end = time.time()
    print('QAOA-M with {} qubits is run in {} seconds'.format(3, end-start))
    durations.append(end-start)

    n = 4  # number of nodes in the graph
    N = n**2 # number of qubits to represent the one-hot-encoding of TSP

    G = nx.Graph()

    for i in range(n):
        G.add_node(i)

    G.add_edge(0,1, weight  = 1)
    G.add_edge(0,2, weight  = 1)
    G.add_edge(1,2, weight  = 100)
    G.add_edge(0,3, weight  = 11)
    G.add_edge(1,2, weight  = 1)
    G.add_edge(1,3, weight  = 100)
    G.add_edge(2,3, weight  = 100)
    edges = list(G.edges)
    start = time.time()
    results.append(qml.specs(circuit_4_mixers_benchmark)(init_params, edges))
    end = time.time()
    print('QAOA-M with {} qubits is run in {} seconds'.format(4, end-start))
    durations.append(end-start)
    
    return results, durations
    