# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:23:54 2022

@author: burak
"""

# understanding qaoa
import pennylane as qml
import numpy as np
import networkx as nx
from scipy.linalg import expm, logm
import time
import qiskit
n = 4 
number_of_qubits = 4
G = nx.Graph()

for i in range(n):
    G.add_node(i)


G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(1,3)
G.add_edge(2,3)
edges = list(G.edges)

nx.draw(G, with_labels=True, alpha=0.8)

# %% 

class CircuitRun:

  def __init__(self, number_of_qubits = 4):
    self.number_of_qubits = number_of_qubits
  

  def edgeCount(self, solution, G):
    edge_count = 0
    edges = G.edges()
    for edge in edges:
      edge_1, edge_2 = edge
      if(solution[edge_1] != solution[edge_2]):
        edge_count += 1
    return edge_count * -1

  def expVal(self, counts, G):
    exp = 0
    total_val = 0
    for sol in counts.items():
      solution, count = sol
      edge_count = self.edgeCount(solution[::-1], G)
      exp += (edge_count * count)
      total_val += count
    return exp/total_val

  def QAOA(self, G, params):
    param_idx = 5 #number of layers
    beta = params[:param_idx]
    gamma = params[param_idx:]
    circuit = qiskit.QuantumCircuit(self.number_of_qubits)
    edge_list = list(G.edges())
    for i in range(self.number_of_qubits):
      circuit.h(i)
    #circuit.barrier()
    for p in range(param_idx):
      for edge in edge_list:
        node_1, node_2 = edge
        #circuit.rz(2 * gamma[p] , node_1)
        #circuit.rz(2 * gamma[p] , node_2)
        circuit.rzz(2 * gamma[p], node_1, node_2)
    #circuit.barrier()
    
      for i in range(self.number_of_qubits):
        circuit.rx(2* beta[p], i)
      
    circuit.measure_all()

    return circuit

  def getExpVal(self, G):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    backend.shots = 8192
    def execute_circ(theta):
      circuit = self.QAOA(G, theta)
      counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
      return self.expVal(counts, G)
    return execute_circ

  def measureCircuit(self, G, theta):
    circuit = self.QAOA(G, theta)
    backend = qiskit.Aer.get_backend('aer_simulator')
    counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
    return counts

  def getResult(self, G, theta):
    counts = self.measureCircuit(G, theta)
    return max(counts, key = counts.get)[::-1], counts # state which has been measured the most frequently

    

QAOA_Circuit = CircuitRun(number_of_qubits)


from qiskit.algorithms.optimizers import SPSA
start = time.time()
optimizer = SPSA(maxiter=1000)
expVal = QAOA_Circuit.getExpVal(G)
optimized_parameters, final_cost, number_of_circuit_calls = optimizer.optimize(8, expVal, initial_point=np.random.rand(10)* np.pi * 2 )
end = time.time()
print('Optimization terminated within {:.3f} seconds' , end-start)
solution,counts = QAOA_Circuit.getResult(G, optimized_parameters)
final_cost

qiskit.visualization.plot_histogram(counts)

#Testing: Classically check number of edges for any possible bitstring

createBitString = lambda x: str(bin(x)[2:].zfill(number_of_qubits))
solution_dictionary = {}

for i in range(2**number_of_qubits):
  solution_dictionary[createBitString(i)] = QAOA_Circuit.edgeCount(createBitString(i), G)
  
  
min_count = 0
min_index = 0
for i in range(2 ** number_of_qubits):
  res = solution_dictionary[createBitString(i)]
  if ( min_count > res ):
    min_count = res
    min_index = i
min_index, min_count



#Test
def checkResult(solution,solution_dictionary, min_count):
  return solution_dictionary[solution] == min_count

 
print(checkResult(solution,solution_dictionary, min_count))

# %% 
N = 4
int2bit = lambda x: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)
quantumOuter = lambda inputs: np.outer(inputs.conj().T, inputs)

z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)


def timeEvolution(local_hamiltonian, psi, timestamp = 1):
    # U = expm(-1j * H * t )
    U = expm(local_hamiltonian * -1j * timestamp)
    return U @ psi

def commuteCheck(A,B):
    return (A@B == B@A).all()
    
def KroneckerProduct(listOfQubits, pauli, N):
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out


def KroneckerProductString(listOfQubits, paulis, N):
    out = np.array([1])
    # idx = 0
    for i in range(N):
        if(i in listOfQubits):
            idx = listOfQubits.index(i)
            out = np.kron(out, paulis[idx])
        else:
            out = np.kron(out, I)
    return out
backend = qiskit.Aer.get_backend('unitary_simulator')
#prepare 2qubits
circ = qiskit.QuantumCircuit(2,2)
beta = (np.pi / 17) * 2
gamma = np.pi/ 1.3
circ.rzz(beta * 2, 0, 1 )
circ.rxx(gamma * 2, 0, 1 )
job = qiskit.execute(circ, backend)
result = job.result()
print(result.get_unitary(circ, decimals=3))
result.get_unitary()


backend = qiskit.Aer.get_backend('statevector_simulator')
unitary_backend = qiskit.BasicAer.get_backend('unitary_simulator')


qasm_backend = qiskit.BasicAer.get_backend('qasm_simulator')

#prepare 2qubits
c = qiskit.ClassicalRegister(4)
q = qiskit.QuantumRegister(4)
circ = qiskit.QuantumCircuit(q,c)

a = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
b = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])

circ.initialize(b , [q[0], q[1], q[2] ,q[3]])

circ.cnot(3,0)
circ.x(3)
circ.cnot(3,2)
circ.cnot(3,1)
circ.x(3)
# circ.reset(3)
# circ.reset(1)
# circ.reset(0)

# qiskit output: 1000 means |0001>

circ.measure(q,c)


qasm_job = qasm_backend.run(qiskit.transpile(circ, qasm_backend))
qasm_result = qasm_job.result()

qasm_result.get_counts()



job = backend.run(qiskit.transpile(circ, backend))
result = job.result()


unitary_job = unitary_backend.run(qiskit.transpile(circ, unitary_backend))
unitary_result = unitary_job.result()

unitary_result.get_unitary(circ, decimals=3) @ a
for i in range(16):
    a = np.zeros(16)
    a[i] = 1
    print(unitary_result.get_unitary(circ, decimals=3) @ a)
print(unitary_result.get_unitary(circ, decimals=3))
result.get_counts()

H = np.array([[1,1] , [1,-1]]) / np.sqrt(2)



# KroneckerProductString([0,1,2,3], [H,H,H,H], 4)



circ.draw()

HC = KroneckerProductString([0,1] , [z,z], 2)
HB = KroneckerProductString([0,1] , [x,x], 2)

evolution = np.round(expm(-1j * HC * beta), 4) @ np.round(expm(-1j * HB * gamma), 4)
gate_unitary = result.get_unitary(circ, decimals=3), 10e-5
np.allclose(evolution, gate_unitary)


dev = qml.device('default.qubit', wires = 7)

def runCircuit(psi):
    res = qCirc(psi)
    return (-1 * res+1) /2

@qml.qnode(dev)
def qCirc(psi):
    qml.QubitStateVector(psi, wires = range(0,4))
    
    qml.IsingZZ(beta * 2, wires = [0,1]).matrix

    
    return [qml.expval(qml.PauliZ(i)) for i in range(9)]
    


# %%
ew, ev = np.linalg.eig(H)
ev[:, 0]
np.max(ew)
H = 0
for edge in edges:
    j,k = edge 
    H += (1/2) * (1 - KroneckerProductString([j,k], [z,z], n))
    
HB = KroneckerProductString([0,1,2], [x,x,x], 3)

psi = np.ones((8,), dtype = np.complex128)
psi /= (2*np.sqrt(2))
expm(-1j * H) @ psi


for i in range(1000):
    psi = expm(-1j * HB* np.pi/3) @ expm(-1j * H* np.pi/3) @ psi
    np.abs(psi)


