# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:52:30 2022

@author: burak
"""

import pennylane as qml

n_qubit_size = 4
latent_size = 1 
n_latent_qubits = latent_size
n_auxillary_qubits = 1
n_total_qubits = n_qubit_size + latent_size + n_auxillary_qubits
dev = qml.device('default.qubit', wires = n_total_qubits )
n_layers = 1



@qml.qnode(dev)
def qCircuit(weights_r, weights_cr, inputs = False):

    qml.QubitStateVector(inputs, wires = range(n_auxillary_qubits + n_latent_qubits, n_total_qubits))
    for l in range(n_layers):                        
        for idx, i in enumerate(range(n_auxillary_qubits + n_latent_qubits, n_total_qubits)):
            qml.Rot(*weights_r[l, 0, idx] , wires = i)
        for idx, i in enumerate(range(n_auxillary_qubits + n_latent_qubits, n_total_qubits)):
            ctr=0
            for jdx, j in enumerate(range(n_auxillary_qubits + n_latent_qubits, n_total_qubits)):
                if(i==j):
                    pass
                else:
                    qml.CRot( *weights_cr[l, idx, ctr], wires= [i, j])
                    ctr += 1
        for idx, i in enumerate(range(n_auxillary_qubits + n_latent_qubits, n_total_qubits)):
            qml.Rot(*weights_r[l, 1, idx] , wires = i)
     
         
         
         
    for i in range(n_auxillary_qubits):
        qml.Hadamard(wires = i)
    for i in range(n_auxillary_qubits):
        qml.CSWAP(wires = [i, i + n_auxillary_qubits , n_auxillary_qubits + i + n_latent_qubits])
    for i in range(n_auxillary_qubits):
        qml.Hadamard(wires = i)
         
         
         
         
         
    return [qml.expval(qml.PauliZ(q)) for q in range(0, n_auxillary_qubits)]



# %% 
n_qubit_size = 2
from DataCreation import dataPreparation
tensorlist = dataPreparation(saved = False, save_tensors = False, method = 'state_prepare', number_of_samples = n_data, number_of_qubits= n_qubit_size)

dev1 = qml.device("default.qubit", wires=n_qubit_size)

inputs = tensorlist[0]
targets =  tensorlist[0]

ketbra = lambda inputs: torch.outer(inputs.conj().T, inputs)
density = ketbra(targets)

ew, ev = np.linalg.eig(density)
ground_state = ev[:, np.argmin(ew)]


@qml.qnode(dev1, interface='torch')
def circuit(params):
    global inputs
    global targets
    global density
    
    cnot_list = []
    
    
    
    qml.QubitStateVector(inputs, wires = range(0, n_qubit_size))
   
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][0], wires=i)
    for idx, i in enumerate(range(n_qubit_size)):
        ctr=0
        for jdx, j in enumerate(range(n_qubit_size)):
            if(i==j):
                pass
            else:
                qml.CRot( *params[i][ctr + 1], wires= [i,j])
                cnot_list.insert(0, [i, ctr+1, i, j])
                ctr += 1
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][n_qubit_size], wires=i)
    
    
    
    for i in range(n_auxillary_qubits):
        qml.Hadamard(wires = i)
    for i in range(n_auxillary_qubits):
        qml.CSWAP(wires = [i, i + n_auxillary_qubits , n_auxillary_qubits + i + n_latent_qubits])
    for i in range(n_auxillary_qubits):
        qml.Hadamard(wires = i)    
    
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][n_qubit_size], wires=i).inv()
    
    for el in (cnot_list):
        i, ctr, i2 , j = el
        qml.CRot( *params[i][ctr], wires= [i2,j]).inv()
        
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][0], wires=i).inv()
        
    return qml.expval(qml.Hermitian(density, wires=list(range(0,n_qubit_size))))

init_params = np.random.rand(n_qubit_size, n_qubit_size + 1,  3)
init_params = torch.from_numpy(init_params)
init_params.requires_grad = True





def isHermitian(H):
    return qml.math.allclose(H ,H.T.conj())
isHermitian(density)

def cost(var):
    return 1-circuit(var)
circuit(var)

res = (cost(init_params))
res.backward()


opt = torch.optim.Adam([init_params], lr = 0.1)

steps = 200

def closure():
    opt.zero_grad()
    loss = cost(init_params)
    print(loss)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)
# %% 
cnot_list = []
for idx, i in enumerate(range(n_qubit_size)):
    
    ctr=0
    for jdx, j in enumerate(range(n_qubit_size)):
        if(i==j):
            pass
        else:
            # qml.CRot( *params[i][ctr + 1], wires= [i,j])
            cnot_list.insert(0, [i, ctr+1, i, j])
            ctr += 1
