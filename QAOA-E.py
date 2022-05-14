# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:58:29 2022

@author: burak
"""

# Libraries
import networkx as nx

import time


import pennylane as qml
from pennylane import numpy as np
from noisyopt import minimizeSPSA
from matplotlib import pyplot as plt
import pickle

from Utils import *
from Ansatz import *
from Visualization.visualizations import *
from Test.benchmark import BenchmarkResults
import subprocess
import os




# %% Experimental setting



n = 3  # number of nodes in the graph
N = n**2 # number of qubits to represent the one-hot-encoding of TSP

G = nx.Graph()

for i in range(n):
    G.add_node(i)

n_wires = N

G.add_edge(0,1, weight  = 10)
G.add_edge(1,2, weight  = 7)
G.add_edge(0,2, weight  = 6)

pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)

with open("data/edges_pickle", "wb") as fp:
    pickle.dump(edges, fp)

np.save('data/adj.npy', adj)

# %% Formulating the Problem Hamiltonian
HC = 0j
l = 0
for edge in edges:
    u, v = edge
    for j in range(n - 1):
        HC += (adj[u, v]) * (KroneckerProduct([(u * n + j), (v * n + j + 1)], z, N) + KroneckerProduct([(u * n + j + 1), (v * n + j)], z, N))
        
    l += (adj[u, v])

l *= (2 - n/2) # normalization
HC += (l * np.eye(2**N))


## Hamiltonian as a pennylane operator
# ops = []
# coeffs = []

# for edge in edges:
#     u, v = edge
#     for j in range(n - 1):
#         op = []
        
#         coeffs.append(adj[u,v])
#         coeffs.append(adj[u,v])
#         ops.append(qml.PauliZ(u*n + j) @ qml.PauliZ(v*n + j + 1))
#         ops.append(qml.PauliZ(u*n + j + 1) @ qml.PauliZ(v*n + j ))
        
# HC = qml.Hamiltonian(coeffs, ops).eigvals

# qml.utils.sparse_hamiltonian(HC).real

# %% QAOA-E Ansatz 
n_layers = 1

ancillary = calculateRequiredAncillaries(n)
dev = qml.device("default.qubit", wires= N + ((N-ancillary) * n_layers), shots=1024)
                
@qml.qnode(dev)
def circuit(params, edge = None, feature_vector = None, test_mode = False):
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
    
    global n_layers, ancillary, edges, adj, n, N, HC
    
    # qml.BasisEmbedding(features=feature_vector, wires=range(9))
    # initialize the qubits in the latent space

    # Initial State Preparation
    for wire in range(0):
        qml.Hadamard(wires=wire)
    
    def reduceFalsePaths():
        qml.Toffoli(wires = [0, 1, 3])
        qml.CNOT(wires = [3, 0])        
        qml.PauliX(3)
        
    reduceFalsePaths()
    # qml.QubitStateVector(statevec, wires=range(9))
    
    U_D(n)
    for l in range(n_layers):
    # Cost unitary
        U_C(params, n, l, edges)
        # Encoder
        U_E(n)
        
        # Discarding the subsystem B, via swapping with a reference system
        discardQubits(n, l)
            
        # Mixer Unitary
        U_M(params, ancillary, n_layers, l)
   
        reduceFalsePaths()
            
        # Decoder, if it is the last QAOA iteration in a test mode, it does not run, since then a state vector in latent
        # space is returned instead of an expectation value of the cost hamiltonian
        if(test_mode == False or (test_mode == True and l != (n_layers - 1))):            
            U_D(n)
        else:
            return qml.probs(range(0,ancillary))
    # return qml.expval(HC)
    return qml.expval(qml.Hermitian(HC, wires=range(N)))


# %% Implementing QAOA-M https://doi.org/10.1145/3149526.3149530

dev_mixer = qml.device("default.qubit", wires= N , shots= 1024)

@qml.qnode(dev_mixer)
def circuit_mixer(params, edge=None, feature_vector = None, test_mode = False):
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
    
    global n_layers
    
    # qml.BasisEmbedding(features=feature_vector, wires=range(9))
    # initialize the qubits in the latent space

    for edge in edges:
        u,v = edge
        for step in range(n - 1):
            
            q_1 = (u * n) + step
            q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
            
            q_1_rev = (u * n) + step + 1 # (reverse path from above)
            q_2_rev = (v * n) + step  

            qml.CNOT(wires=[q_1, q_2])
            qml.RZ(params[1], wires= q_2)
            qml.CNOT(wires=[q_1, q_2])
            
            qml.CNOT(wires=[q_1_rev, q_2_rev])
            qml.RZ(params[1], wires= q_2_rev)
            qml.CNOT(wires=[q_1_rev, q_2_rev])    
            
   
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
                            
    if(test_mode == True):
        
        return qml.probs(range(9))
    
    return qml.expval(qml.Hermitian(HC, wires=[0,1,2,3,4,5,6,7,8]))

# %% Training with SPSA 

lr = 0.03 # for ADAM and RMSProp

losses = []
training_routines = []
preds = []
running_avgs = []
accuracies = []
opt_params = []

from pennylane import numpy as np

flat_shape = n_layers * 2
param_shape = (n_layers * 2, 1)
initial_params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)
initial_params = initial_params.reshape(flat_shape)
initial_params[0] = -4
initial_params[1] = -1
# %% 
niter_spsa = 10

circuits = [circuit, circuit_mixer]
devs = [dev, dev_mixer]

for idx, circuit_exp in enumerate(circuits):
    dev_exp = devs[idx]
    
    device_execs = []
    spsa_losses = []
    spsa_preds = []
    
    init_params = initial_params.copy()

    init_params[0] = -4 # global minima, the initial parameter is set that way, since the training is trivial otherwise
    init_params[1] = -1

    ctr = 0

    current_model = circuit_exp
    current_dev = dev_exp

    def callback_fn(params):
        global start, ctr
        
        loss = current_model(params)
        if(idx == 1):
            print(loss)
        pred = current_model(params, test_mode = True)
        
        spsa_preds.append(pred)
        spsa_losses.append(loss)
        
        num_executions = int(dev_exp.num_executions / 2)
        device_execs.append(num_executions)
        end = time.time()
        
        ctr += 1
        print('Epoch {} elapsed in {}s,  Loss: {}'.format(ctr , end - start, loss))
    start = time.time()
    
    res = minimizeSPSA(
        current_model,
        x0=init_params.copy(),
        niter=niter_spsa,
        paired=False,
        c=0.15,
        a=0.2,
        callback = callback_fn
    )
    # SPSA outputs
    
    losses.append(spsa_losses)
    preds.append(spsa_preds)
    opt_params.append(res.x)
    spsa_running_avgs = calculateRunningAverage(spsa_losses)
    running_avgs.append(spsa_running_avgs)
    spsa_accuracies = []
    for i in range(len(spsa_preds)):
        spsa_accuracies.append(validationError(spsa_preds[i]))
    accuracies.append(spsa_accuracies)
    if(idx == 0):
        routine = 'SPSA for QAOA-E'
    else:
        routine = 'SPSA for QAOA-M'
    training_routines.append(routine)
# %% Convergence plot between QAOA-E and QAOA-M using the SPSA

loss_plot(running_avgs, training_routines, 'loss')



# %% SGD

rmsOptimizer = qml.RMSPropOptimizer(stepsize = lr)
adamOptimizer = qml.AdamOptimizer(stepsize = lr)
rmsOptimizer_mixer = qml.RMSPropOptimizer(stepsize = lr)
adamOptimizer_mixer = qml.AdamOptimizer(stepsize = lr)
opts = [rmsOptimizer, adamOptimizer, rmsOptimizer_mixer, adamOptimizer_mixer]
opt_names = ['RMSProp with QAOA-E' , 'ADAM with QAOA-E' , 'RMSProp with QAOA-M' , 'ADAM with QAOA-M' ]
epochs = 3
circuits = [circuit, circuit, circuit_mixer, circuit_mixer]
for idx, opt in enumerate(opts):
    init_params = initial_params.copy()
    cur_losses = []
    cur_preds =  []
    cur_accs =   []
    circuit_exp = circuits[idx]
    for i in range(epochs):
        start = time.time()
        init_params = opt.step(circuit_exp, init_params)
        
        loss = circuit_exp(init_params)
        pred = (circuit_exp(init_params, test_mode = True))
        
        cur_preds.append(validationError(pred))
        cur_losses.append(loss)
        if(i % 10):
            end =  time.time()
            print('Epoch {} elapsed in {}s,  Loss: {}'.format( i , end - start, loss))
    
    running_avg = calculateRunningAverage(cur_losses)
    running_avgs.append(running_avg)
    losses.append(cur_losses)
    
    accuracies.append(cur_preds)
    opt_params.append(init_params)
    training_routines.append(opt_names[idx])

# %%  Save Results

np.save('data/opt_params.npy', opt_params)


# %% Visualization

loss_plot(losses, training_routines, 'loss')
loss_plot(accuracies, training_routines, 'acc')


# %% Energy Landscape

search_params = initial_params.copy()
param_interval = 0.25
epsilon= np.arange(-np.pi, np.pi, param_interval)

# %% Visualize Energy Landscapes

objective_fn_evals = np.zeros((epsilon.shape[0], epsilon.shape[0] ))

for idx1, epsilon1 in enumerate(epsilon):
    for idx2, epsilon2 in enumerate(epsilon):
        search_params[0] = epsilon1
        search_params[1] = epsilon2
        objective_fn_evals[idx1,idx2] = circuit(search_params)
        
energyLandscape(epsilon, objective_fn_evals)


# %% QAOA-M Training with Adam and RMSProp

mixer_params = initial_params.copy()

mixer_losses = []
mixer_preds = []
mixer_opt = qml.RMSPropOptimizer(stepsize = 0.03)
for i in range(3):
    start = time.time()
    mixer_params = mixer_opt.step(circuit_mixer, mixer_params)
    
    mixer_loss = circuit(mixer_params)
    print(mixer_loss)
    mixer_pred = (circuit(mixer_params, test_mode = True))
    
    mixer_preds.append(mixer_pred)
    mixer_losses.append(mixer_loss)
    end = time.time()
    if(i % 10):
        print('Epoch {} elapsed in {}s,  Loss: {}'.format( i , end - start, mixer_loss))

# %% Circuits Specs for benchmarking

results, durations = BenchmarkResults()

# Run Unit Tests
execfile('Test//unitTest.py')

# %% Plot the depth

DepthLimit()
