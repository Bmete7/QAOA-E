# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:03:14 2022

@author: burak
"""

from pennylane import numpy as np
import pennylane as qml
import numpy
import scipy as sp
import torch
import itertools

# Pauli Matrices

z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)

# Annihilation, Creation Operators
Splus = x + 1j*y
Sminus = x - 1j*y

createBitString = lambda x,y=9: str(bin(x)[2:].zfill(y))
int2bit = lambda x,y=9: str(bin(x)[2:].zfill(y))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2) 
quantumOuter = lambda i1,i2: np.outer(i1, i2.conj().T) # braket
norm = lambda x:np.linalg.norm(x)
normalize = lambda x: x/norm(x)

def stateVectorCheck(psi):
    '''
    Checks if a vector is a valid state vector
    by checking if all measurement possibilities for each state adds up to 1 or not
    
    Parameters
    ----------
    psi : Complex vector

    Returns
    -------
    bool
    '''
    return np.isclose(np.sum(np.abs(psi)**2), 1, 10e-4)
    


def commuteCheck(A, B):
    '''
    Checks if two matrices A and B commute
    Parameters
    ----------
    A : matrix, 2D array
    B : matrix, 2D array

    Returns
    -------
    Bool
    '''
    return np.allclose(A@B, B@A, 10e-3)
    
def KroneckerProduct(listOfQubits, pauli, N):
    '''
    Returns an operator that a Pauli String Forms. Note that it only accepts
    one Pauli type. For a generalized version, see KroneckerProductString 

    Parameters
    ----------
    listOfQubits : indices for the Pauli String
    pauli : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    '''
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out

def sparseDiag(diags):
    '''
    Given an array, returns a diagonal sparse matrix

    Parameters
    ----------
    array : List of elements in the diagonals

    Returns
    -------
    scipy.sparse matrix

    '''
    return sp.sparse.diag(diags)
    

def efficientDiagonalKronecker(listOfQubits, pauli, N):
    '''
    Generates a vector, whose diagonalized version is a  Kronecker Product of Pauli-Z and Identities

    Parameters
    ----------
    listOfQubits : list
        Indices for PauliZ operators
    pauli : pauli gate, should be pauli-z by default
    N : int
        Total number of qubits
    Returns
    -------
    dummy : TYPE
        DESCRIPTION.

    '''
    dummy = np.ones(2**N)
    for l in listOfQubits:
        order = 2**(N-l)
        for i in range(len(dummy)):
            if(i % order >= (order / 2)):
                dummy[i] *= -1
    return dummy

def KroneckerProductString(listOfQubits, paulis, N):
    '''
    Given a Pauli String, return the Kronecker product for N qubits
    i.e (1,2) (Z,Z) (5) = I Z Z I I 

    Parameters
    ----------
    listOfQubits : list
        Qubit lines where the operators apply
    paulis : Square matrix
        The operator
    N : int
        Number of qubit lines

    Returns
    -------
    out : matrix
        The kronecker product

    '''
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            idx = listOfQubits.index(i)
            out = np.kron(out, paulis[idx])
        else:
            out = np.kron(out, I)
    return out


def findGroundStates(H, size = 9):
    '''

    Parameters
    ----------
    H : Hamiltonian

    Returns
    -------
    ground_state : Ground state vector
    lowest_energy_states : lowest eigenvalue index
    lowest_energy : lowest eigenvalue

    '''
    ew , ev = np.linalg.eig(H)
    
    lowest_energy_state = np.argmin(ew)
    ground_state = ev[:, lowest_energy_state]
    solutions = np.where(np.abs(ground_state) > 10e-2)
    
    return [createBitString(solution, size) for solution in solutions[0]], ground_state, solutions, lowest_energy_state



def checkConstraints(solution, n = 3):
    '''
    Given a one-hot-encoding, find the edges in a solut

    Parameters
    ----------
    solution : str
        One-hot encoded solution
    n : int
        number of vertices in the graph

    Returns
    -------
    bool, True if the solution satisfies the constraints

    '''
    
    for q in range(n):
        for i in range(n):
            for j in range(n):
                if(i != j):
                    u = q * n + i
                    v = q * n + j
                    if( solution[u] == '1' and solution[v] == '1'):
                        return False
    for j in range(n):
        for q1 in range(n):
            for q2 in range(n):
                if(q1 == q2):
                    continue
                u = q1 * n + j
                v = q2 * n + j
                if( solution[u] == '1' and solution[v] == '1'):
                    return False
                
    return sum([int(el) for el in solution]) == n



def testCheckConstraints(n = 3):
    '''
    Unit test for checkConstraint method

    Parameters
    ----------
    n : int
        number of vertices in the graph

    Returns
    -------
    bool

    '''
    ground_truth = [84,98,140,161,266,273]
    sol = []
    for i in range(2**(n**2)):
        if(checkConstraints(int2bit(i), n)):
            sol.append(i)
    
    return sol == ground_truth


def findPath(solution, n = 3):
    if(checkConstraints(solution) == False):
        raise('Faulty encoding')
    path = np.array([0,0,0])
    for i in range(n):
        for j in range(n):
            
            if(solution[i * n + j] == '1' ):
                path[j] = i
                continue
    return path


def guessState(out):
    '''
    Predicting a route, depending on the state vector in the latent space

    Parameters
    ----------
    out : State vector of the latent space 

    Returns
    -------
    TYPE
        Bitstring of the predicted route

    '''
    majority_vote = np.argmax(out[:-2])
    return int2bit(majority_vote, 3)


def validationError(pred):
    # calculate the accuracy w.r.t costly paths. 
    # Disregard the optimal paths.(since they are always multiple)
    error = 0
    ground_truth = [0,0,1,1,0,0,0,0]
    for i in range(len(ground_truth)):
        if(i== 2 or i == 3):
            continue
        else:
            error += np.abs(ground_truth[i] - pred[i])
    return 1 - error


def calculateRunningAverage(losses):
    '''
    Given a list of losses, calculate the running average of the loss

    Parameters
    ----------
    losses : list

    Returns
    -------
    running_avg: list

    '''
    running_avg = []
    running_loss = 0
    for idx,loss in enumerate(losses):
        running_loss += loss
        running_avg.append(running_loss / (idx + 1))
    return running_avg




def calculateRequiredAncillaries(n):
    '''
    Given a vertex number, this method returns the number of required
    ancillary qubits to solve a constraint satisfaction problem

    Parameters
    ----------
    n : int
        

    Returns
    -------
    n_ancillary: int

    '''
    if(n <= 2):
        return 1
    return int(np.ceil(np.log(numpy.math.factorial((n - 1)) * (n - 2))) + n - 1)





class HamiltonianUtils():
    '''
    Utility Functions for QAutoencoder and Hamiltonian Simulator
    
    Attributes
    ----------
    dev:
        quantum device object
    
    
    Methods
    ----------
    
    returnState(params):
        returns the state vector for a given sequence
        
    '''
    def __init__(self, dev, n_qubit_size = 8):
        self.dev = dev
        self.n_qubit_size = n_qubit_size
        self.torchnorm = lambda x:torch.norm(x)
        self.torchnormalize = lambda x: x/self.torchnorm(x)
        self.quantumOuter = lambda inputs: torch.outer(inputs.conj().T, inputs)
        self.wv = torch.ones(4, dtype = torch.cdouble) / torch.sqrt(torch.Tensor([4]))
        
        @qml.qnode(self.dev)
        def returnState(psi):
            qml.QubitStateVector(psi, wires = range(0, self.n_qubit_size))
            return qml.state()
        
        self.ReturnState = returnState
    def getState(self, psi):
        return self.ReturnState(psi)
    
    def createHamiltonian(self, edges, n,  pauli = 'Z'):
        pauli_Z = np.array([[1,0] , [0j, -1]])
        pauli_Y = np.array([[0,-1j] , [1j, 0]])
        pauli_X = np.array([[0,1] , [0j + 1, 0]])
        H_locals = []
        H_final = 0
        started = False
        def createLocalHamiltonian(i,j, pauli = pauli):
            emp= np.array([1])
            for k in range(n):
                if(k==i or k == j):
                    emp = np.kron(emp,pauli_Z)
                else:
                    emp = np.kron(emp,np.eye(2))
            return emp
        num_of_hamiltonians = len(edges)
        for edge in edges:
            H_local = createLocalHamiltonian(edge[0] , edge[1] ) / np.sqrt(num_of_hamiltonians)
            if(started == False):
                started = True
                H_final = H_local * 1
            else:
                H_final = H_final + H_local

        return H_final   
    
    
    def findHermitian(self, coeffs, randomPauliGroup, n_qubit_size):
        '''
        Given Pauli strings, it prepares an arbitrary Hamiltonian

        Parameters
        ----------
        coeffs : list
            List of pauli coefficients
        randomPauliGroup : list
            List of Pauli strings
        n_qubit_size : TYPE
            Number of qubits

        Returns
        -------
        hamiltonian : np.ndarray
            Hermitian matrix

        '''
        I = np.eye(2)
        pauliSet = [qml.PauliX(0).matrix, qml.PauliZ(0).matrix, qml.PauliY(0).matrix]
        hamiltonian = 0j
        for i in range(n_qubit_size):
            pauli_word = np.array([1])
            cur_hermitian = pauliSet[randomPauliGroup[i]] * coeffs[i]
            for j in range(n_qubit_size):
                if(i==j):
                    pauli_word = np.kron(pauli_word, cur_hermitian)
                else:
                    pauli_word = np.kron(pauli_word, I)
                    
            hamiltonian += pauli_word
        return hamiltonian



    def timeEvolution(self, local_hamiltonian, psi, timestamp):
        # U = expm(-1j * H * t )
        U = torch.matrix_exp(local_hamiltonian * -1j * timestamp)
        
        return U @ psi
    
    def decompose_hamiltonian(self, hamiltonian,paulis):
        """ Finds the coefficients for the product state of pauli matrices that
        composes a hamiltonian, from a given hermitian

        Non-Keyword arguments:
            Hamiltonian- Hermitian matrix
        Keyword arguments:
            paulis: Pauli matrices with the given order: I,pX,pY,pZ
            
        Returns:
            coefs: Coefficients of pauli matrices, a1 III + a2 IIX + .. an ZZZ 
        """
        num_el = np.prod(list(hamiltonian.shape)) # number of elements of the hamiltonain
        assert np.log2(num_el) % 1 == 0, 'Hamiltonian is not a valid hermitian'
        
        n_qubits = np.log2(num_el) / 2
        assert n_qubits % 1 == 0, 'Hamiltonian is not a valid hermitian'
        
        
        n_qubits = int(n_qubits)
        num_el = int(num_el)
        
        pauli_prods = self.pauli_product_states(n_qubits, paulis)
        coefs = np.zeros(num_el , dtype = 'complex128')
        for i,prod in enumerate(pauli_prods):
            coefs[i] = np.trace(pauli_prods[i] @ hamiltonian) / 4
        
        for i in range(len(coefs)):
            assert coefs[i].imag == 0.0, 'Complex parameters are found in the coefficients, is the Hamiltonian a Hermitian matrix?'
        
        actual_coefs = np.zeros(num_el , dtype = 'float64')
        for i in range(len(coefs)):
            actual_coefs[i] = coefs[i].real
        return actual_coefs
        
    def pauli_product_states(self, n_qubits,paulis):
        
        all_paulis = []
        
        for i in range(n_qubits):
            all_paulis.append(paulis)
        
        
        pauli_prods = []
        for combination in itertools.product(*all_paulis):
            
            prod_ind = combination[0]
            for i in range(1 , len(combination)):
                prod_ind  = np.kron(prod_ind , combination[i])
            pauli_prods.append(prod_ind)
        
        return pauli_prods


