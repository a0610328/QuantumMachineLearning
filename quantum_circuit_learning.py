# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:04:58 2021

@author: Huijie Guan

Pick More Daisies

FYI, with 40 training data, 100 max iteration steps, the training 
process took ~ 3 hours, accuracy 0.475. Accuracy 20 Test datas are 0.6
The randomly generated training data is less separable than that of 
the test data, which may account for the increase in accuracy. However,
there exists a definite boundary between the two categories. The
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from qiskit.opflow import StateFn
from qiskit import QuantumCircuit
from qiskit.opflow import X,Z,I, CircuitOp
from itertools import permutations
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import ADAM
import qiskit.utils.algorithm_globals
import matplotlib.pyplot as plt
from time import time
from random import seed


def circ_fm(n_qubits, x):
    """
    
    Parameters
    ----------
    n_qubits : int, number of qubits
    x: 1d ndarray, float, (2,)

    Returns
    -------
    circ_fm: qiskit circuit with n_qubits bits with unbound parameters x
    |s\rangle = prod_j R_j^Z(cos^{-1} x^2_{j mod 2}) R_j^Y(sin^{-1} x_{j mod 2}) 

    """
    assert all(abs(x)<=1)
    phasez1 = np.arccos(x[0]**2)
    phasez2 = np.arccos(x[1]**2)
    phasey1 = np.arcsin(x[0])
    phasey2 = np.arcsin(x[1])
    qc_fm = QuantumCircuit(n_qubits)
    for i in np.arange(start = 0, stop = n_qubits, step = 2):
        qc_fm.rz(phasez1, i)
        qc_fm.ry(phasey1, i)
        qc_fm.rz(phasez2, i+1)
        qc_fm.ry(phasey2, i+1)
    return qc_fm
    
def circ_ansatz_ising(n_qubits, T):
    """
    
    Parameters
    ----------
    n_qubits : int
    T: float, duration for Ising evolution
        
    Returns
    -------
    HT : PauliSumOp, T* H_{ising} which will later be exponentiated
        as exp(-iHT). Ising Hamiltonian H is defined as 
                H = \sum_j a_j X_j + \sum_j\sum_k J_{jk} Z_j Z_k
        with a_j, J_{jk} taken randomly from unifrom distribution

    """
    
    a = T * np.random.uniform(size = n_qubits)
    J = T * np.random.uniform(size = int(n_qubits*(n_qubits-1)/2))
    #TODO: find more elegant way to implement the Hamiltonian. So far, we have
    #to create a string like '(0.1*Z^I^Z^I)+(0.2^I^Z^I^Z)' and use eval() to create
    #a PauliSumOp
    # Create a list of X_terms like 'X^I^I', 'I^X^I', 'I^I^X'
    x_term = ''.join(['X']+['I']*(n_qubits-1))
    X_list =  list(set(['^'.join(x_str) for x_str in permutations(x_term)]))
    # Then add coefficients to each term in a string, then add them up in a string
    X_exp = '+'.join(f'({str(a_i)}*{x_i})' for a_i, x_i in zip(a, X_list))
    # Create a list of Z_term like 'Z^Z^I^I', 'Z^I^Z^I'
    z_term = ''.join(['ZZ']+['I']*(n_qubits-2))
    Z_list =  list(set(['^'.join(z_str) for z_str in permutations(z_term)]))
    # Then add coefficients to each term in a string, then add them up in a string
    Z_exp = '+'.join(f'({str(j_i)}*{z_i})' for j_i, z_i in zip(J, Z_list))
    TH = eval(X_exp +'+'+ Z_exp)
    #print(X_exp + '+' + Z_exp)
    return TH

    
def circ_ansatz_unitary(n_qubits, layer):
    """
    Parameters
    ----------
    n_qubits : int
    
    Returns
    -------
    U_ansatz : CircuitOp instance, variational ansatz with parameter theta_1, 
        ..., theta_{n_qubits}, defined as 
            prod_i R_i^X(theta_1^i) R_i^Z(\theta_2^i) R_i^X(\theta_3^i)
    """
    U_ansatz = QuantumCircuit(n_qubits)
    theta = ParameterVector(f'theta_{layer}', 3*n_qubits)
    for i in range(n_qubits):
        U_ansatz.rx(theta[3*i], i)
        U_ansatz.rz(theta[3*i+1], i)
        U_ansatz.rx(theta[3*i+2], i)
    return CircuitOp(U_ansatz)
    
def build_circ(n_qubits, D = 1):
    """
    
    Parameters
    ----------
    n_qubits : integer, number of qubits
    D: integer, number of repetitions in variational ansatz
    Returns
    -------
    qc:  ComposedOp instance, defined as
        prod_{i}^D (e^{-i H_{ising} T} prod_j prod_i R_i^X(theta_1^i) R_i^Z(\theta_2^i) R_i^X(\theta_3^i))

    """ 
    # U is an umparametrized CircuitOp instance 
    U = circ_ansatz_unitary(n_qubits, layer = 0)
    qc = U @ qc_ising
    #qc_block.to_circuit().draw()
    #qc = qc_block
    for i in range(D-1):
        U = circ_ansatz_unitary(n_qubits, layer = i+1)
        qc_block = U @ qc_ising
        qc = qc_block @ qc
    return qc

def predictions(X_dat, theta):
    """
    
    Parameters
    ----------
    X: 2d ndarray, float, (rows, 2)
    theta : 1d ndarray, variational parameter (3*n_qubits)

    Returns
    -------
    pred_list: list of (z1, z2), (rows, )
    z1: float, <\psi(x, theta) |sigma_z^1|\psi(x,theta)>
    z2: float, <\psi(x, theta) |sigma_z^2|\psi(x,theta)>

    """
    
    # quantum circuit
    param_dict = {}
    for p_i, theta_i in zip(qc.parameters, theta):
        param_dict[p_i] = theta_i
    qc_w_parameter = qc.assign_parameters(param_dict)
    
    # observables 'ZIII', 'IZII'
    z1_op = eval('Z'+'^I'*(n_qubits-1))
    z2_op = eval('I'+'^Z'+'^I'*(n_qubits-2))
    
    pred_list = []
    for x in X_dat:
        # initial state encoding x
        fm = circ_fm(n_qubits, x)
        state_init = StateFn(fm)
    
        z1 = (~state_init @ ~qc_w_parameter @ z1_op @ qc_w_parameter\
              @ state_init).eval().real
        z2 = (~state_init @ ~qc_w_parameter @ z2_op @ qc_w_parameter \
              @ state_init).eval().real
        pred_list.append((z1, z2))
        
    return np.array(pred_list)
def softmax(v):
    """
    
    Parameters
    ----------
    v : 1d ndarray

    Returns
    -------
    vv: 1d ndarray, vv_i = np.exp(v_i)/sum_i (np.exp(v_i))

    """
    numerator = list(map(np.exp, v))
    denominator = np.sum(numerator)
    return numerator/denominator


def loss(X_dat, Y_dat, theta):
    """
    

    Parameters
    ----------
    X : 2d ndarray, float, (rows, 2)
    Y : 1d ndarray, integer, (rows,)
    theta : 1d ndarray, float, (3*n_qubits,)

    Returns
    -------
    l: float, averaged cross entropy loss, for each data, loss entropy is 
        defined as -ylog(softmax(z1))-(1-y)log(softmax(z2))    

    """   
        
        
    l = 0
    pred_list = predictions(X_dat, theta)
    for (z1, z2), y in zip(pred_list, Y_dat):
        zz1, zz2 = softmax((z1, z2))
        l += -(1-y)*np.log(zz1)-y*np.log(zz2)  #TODO check this
    return l/len(Y_dat)
    

def gradient(X_dat, Y_dat, theta, loss_track, accu_track, minibatch_size = None):
    """
    Parameters
    ----------
    X_dat: 2d ndarray, float, input features, (rows, 2)
    Y_dat: 1d ndarray, int, true lables, (rows, )
    theta: 1d ndarray, float, to be optimized parameter in unitary circuits, (3*n_qubits*D)
    loss_track: list, float, record of loss function 
    Returns
    -------
    grad: 1d ndarray, float, dloss/dtheta_i, (3*n_qubits*D,)
        -y (1-softmax(x_1))\partial x_1/\partial theta_i -(1-y)(1-softmax(x_2))\
            \partial x_2/\partial theta_i

    """  
    if minibatch_size:
        minibatch_index = np.random.choice(np.arange(len(Y_dat)), minibatch_size)
        X_minibatch = np.array([X_dat[i] for i in minibatch_index])
        Y_minibatch = np.array([Y_dat[i] for i in minibatch_index])
    else:
        X_minibatch, Y_minibatch = X_dat, Y_dat
        
    loss_track.append(loss(X_dat, Y_dat, theta))
    accu_track.append(accuracy(X_dat, Y_dat, theta))
    plt.plot(range(len(loss_track)), loss_track)
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy Loss Function')
    plt.show()
    print(f'accuarcy so far is {accu_track}')
             
    grad_list = []
    dtheta = np.pi/2* np.eye(len(theta))
    pred_list_theta = predictions(X_minibatch, theta)
    pred_list_theta_softmax = np.array(list(map(softmax, pred_list_theta)))
    for dt in dtheta:
        pred_list_plus = predictions(X_minibatch, theta+dt)
        pred_list_minus = predictions(X_minibatch, theta-dt)
        pred_list_dt = (pred_list_plus-pred_list_minus)*0.5
        grad_list_theta = -Y_minibatch * (1-pred_list_theta_softmax[:,1])*\
            pred_list_dt[:,1] -(1-Y_minibatch)*(1-pred_list_theta_softmax[:,0])\
                *pred_list_dt[:,0]
        grad_list.append(np.sum(grad_list_theta, axis = 0)/minibatch_size)
    return np.array(grad_list)
        
        
def optimize(X_dat, Y_dat, minibatch_size = None, maxiter = 50):
    #loss_track keeps record of the loss function during the training process
    loss_track = []
    accu_track = []
    theta_init = np.random.uniform(size = 3*n_qubits*D)
    objective_function = lambda params: loss(X_dat, Y_dat, params)
    gradient_function = lambda params: gradient(X_dat, Y_dat, params, loss_track, \
                                                accu_track, minibatch_size)
    maxiter = 100  #TODO maxiter = 100
    tol = 1e-3
    lr = 2.5* 1e-2 
    num_vars = int(3*n_qubits*D)
    optimizer = ADAM(maxiter = maxiter, tol = tol, lr = lr)
    point, value, _ = optimizer.optimize(num_vars = num_vars, objective_function \
                                         = objective_function, gradient_function \
                                             = gradient_function, initial_point = theta_init)
    return point, value
    
    
def accuracy(X_dat, Y_dat, theta):
    """

    Parameters
    ----------
    X_dat : 2d ndarray, float, (rows, 2)
    Y_dat : 1d ndarray, int, (rows,)
    theta: 1d ndarray, float, (3*n_qubits*D, )

    Returns
    -------
    accuracy: float, average accuracy

    """
    pred = predictions(X_dat, theta)
    accu_count = 0
    for (z1, z2), y in zip(pred, Y_dat):
        if (z2-z1)*(y-0.5)> 0:
            accu_count +=1
    return accu_count/len(Y_dat)
    
def get_data(n_samples = 200, plot = True):
    """
    
    Parameters
    ----------
    n_samples : integer, total number of datas, with split equally between two species
    plot : boolean, if true, plot the data

    Returns
    -------
    X : 2d ndarray, float,(n_samples, 2)
    Y : 1d ndarray, int, (n_sample,)

    """
    X, Y = datasets.make_circles(n_samples = n_samples, noise = 0.1, \
                                 factor = 0.3, random_state= 1)
    X_scale = np.max(np.abs(X), axis = 0)
    X = X/X_scale
    if plot:
        red = (Y == 0)
        blue = (Y == 1)
        plt.scatter(X[red, 0], X[red, 1], c = 'red')
        plt.scatter(X[blue, 0], X[blue, 1], c = 'blue')
        plt.show()
    return X,Y


# circuit parameters
n_qubits = 4
T = 10 
D = 2

# Unparametrized circuit part
# Ising Hamiltonian times T,  a PauliSumOp
th = circ_ansatz_ising(n_qubits, T)
# Exponential of (Ising Hamiltonian times T), an EvolvedOp
qc_ising = th.exp_i()
# qc combines ising and unparametrized unitary circuit
qc = build_circ(n_qubits, 1)

# Get training data
n_samples = 20 
X_dat, Y_dat = get_data(n_samples = n_samples, plot = True) 
#theta = np.random.uniform(size = 3*n_qubits)
#g = gradient(X_dat, Y_dat, theta)

# Training

t1 = time()
minibatch_size = 20
maxiter = 50
qiskit.utils.algorithm_globals.random_seed = 10
print(f'Training started for n_qubits = {n_qubits}, n_sample = {len(Y_dat)}, minibatch size = {minibatch_size}, maxiter = {maxiter}')
theta, _ = optimize(X_dat, Y_dat, minibatch_size, maxiter)
t2 = time()
print(f'training takes {t2-t1} time')
accu_train = accuracy(X_dat, Y_dat, theta)
print(f'accuracy for training data is {accu_train}')


#Testing  
X_test, Y_test = get_data(n_samples = 20, plot = True) 
accu_test = accuracy(X_test, Y_test, theta)
print(f'accuracy on test data is {accu_test}')

#Plot prediction
X_test_plot = np.array([[(i,j) for i in np.arange(-1, 1, 0.2)] for j in np.arange(-1,1,0.2)])
Y_test_plot = np.zeros(X_test_plot.shape[:2])
for j in range(10):
    pred = predictions(X_test_plot[:,j].reshape(-1,2), theta)
    Y_test_plot[:,j] = np.array(list(map(softmax, pred)))[:,0]
fig, (ax1, ax2) = plt.subplots(1,2)
img1 = ax1.imshow(Y_test_plot, vmin = 0, vmax = 1)
plt.colorbar(img1, ax = ax1)
img2 = ax2.contourf(Y_test_plot, levels = [0,0.5,1], colors = ['red', 'purple'], alpha = 0.8)
plt.colorbar(img2, ax = ax2)
plt.axis('square')
plt.show()