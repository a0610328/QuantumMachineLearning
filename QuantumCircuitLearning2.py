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
#from qiskit.algorithms.optimizers import ADAM, Nel
import matplotlib.pyplot as plt
from time import time
import os
import pickle


class QuantumCircuitLearning():
    
    def __init__(self, n_qubits, T, D = 1):
        self.n_qubits = n_qubits
        self.T = T
        self.D = D
        self.qc_ansatz = self.build_ansatz()
# =============================================================================
#         self.qc_fm
#         self.qc_ising
# =============================================================================
        
    def circ_fm(self, x):
        """
        generate self.qc as a qiskit circuit with n_qubits bits with unbound parameters x
        |s\rangle = prod_j R_j^Z(cos^{-1} x^2_{j mod 2}) R_j^Y(sin^{-1} x_{j mod 2}) 
        
        Parameters
        ----------
        x: 1d ndarray, float, (2,)
          
    
        """
        assert all(abs(x)<=1)
        phasez1 = np.arccos(x[0]**2)
        phasez2 = np.arccos(x[1]**2)
        phasey1 = np.arcsin(x[0])
        phasey2 = np.arcsin(x[1])
        qc_fm = QuantumCircuit(self.n_qubits)
        for i in np.arange(start = 0, stop = self.n_qubits, step = 2):
            qc_fm.rz(phasez1, i)
            qc_fm.ry(phasey1, i)
            qc_fm.rz(phasez2, i+1)
            qc_fm.ry(phasey2, i+1)
        self.qc_fm = qc_fm
        
    def circ_ansatz_ising(self):
        """
        generate self.HT : PauliSumOp, T* H_{ising} which will later be exponentiated
            as exp(-iHT). Ising Hamiltonian H is defined as 
                    H = \sum_j a_j X_j + \sum_j\sum_k J_{jk} Z_j Z_k
            with a_j, J_{jk} taken randomly from unifrom distribution
            
        
        Parameters
        ----------
        n_qubits : int
        T: float, duration for Ising evolution
            
        """
        
        a = T * np.random.uniform(size = self.n_qubits)
        J = T * np.random.uniform(size = int(self.n_qubits*(self.n_qubits-1)/2))
        #TODO: find more elegant way to implement the Hamiltonian. So far, we have
        #to create a string like '(0.1*Z^I^Z^I)+(0.2^I^Z^I^Z)' and use eval() to create
        #a PauliSumOp
        # Create a list of X_terms like 'X^I^I', 'I^X^I', 'I^I^X'
        x_term = ''.join(['X']+['I']*(self.n_qubits-1))
        X_list =  list(set(['^'.join(x_str) for x_str in permutations(x_term)]))
        # Then add coefficients to each term in a string, then add them up in a string
        X_exp = '+'.join(f'({str(a_i)}*{x_i})' for a_i, x_i in zip(a, X_list))
        # Create a list of Z_term like 'Z^Z^I^I', 'Z^I^Z^I'
        z_term = ''.join(['ZZ']+['I']*(self.n_qubits-2))
        Z_list =  list(set(['^'.join(z_str) for z_str in permutations(z_term)]))
        # Then add coefficients to each term in a string, then add them up in a string
        Z_exp = '+'.join(f'({str(j_i)}*{z_i})' for j_i, z_i in zip(J, Z_list))
        TH = eval(X_exp +'+'+ Z_exp)
        #print(X_exp + '+' + Z_exp)
        self.qc_ising = TH
    
        
    def circ_ansatz_unitary(self, layer):
        
        """
                
        Parameters
        ----------
        layer: integer, the label of the layer of the unitary operation, must be
            less than self.D
            
        Returns:
        U: a CircuitOp instance, variational ansatz with parameter theta_1, 
            ..., theta_{n_qubits}, defined as 
                prod_i R_i^X(theta_1^i) R_i^Z(\theta_2^i) R_i^X(\theta_3^i)    
        
        """
        U_ansatz = QuantumCircuit(self.n_qubits)
        theta = ParameterVector(f'theta_{layer}', 3*self.n_qubits)
        for i in range(self.n_qubits):
            U_ansatz.rx(theta[3*i], i)
            U_ansatz.rz(theta[3*i+1], i)
            U_ansatz.rx(theta[3*i+2], i)
        U = CircuitOp(U_ansatz)
        
        return U
        
    def build_ansatz(self):
        """
        set self.qc, a ComposedOp instance, defined as
            prod_{i}^D (e^{-i H_{ising} T} prod_j prod_i R_i^X(theta_1^i) R_i^Z(\theta_2^i) R_i^X(\theta_3^i))
            
        Parameters
        ----------
       
    
        """ 
        # set self.qc_ising, a PauliSumOp of evolution under ising hamiltonian
        self.circ_ansatz_ising()


        # set self.qc_ansatz
        U = self.circ_ansatz_unitary(layer = 0) 
        qc =  U @ self.qc_ising
        for i in range(D-1):
            U = self.circ_ansatz_unitary(layer = i+1)
            qc_block = U @ self.qc_ising
            qc = qc_block @ qc
        return qc
    
    def predictions(self, X_dat, theta):
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
        
        
        # assign parameters
        param_dict = {}
        for p_i, theta_i in zip(self.qc_ansatz.parameters, theta):
            param_dict[p_i] = theta_i
        qc_w_parameter = self.qc_ansatz.assign_parameters(param_dict)
        
        # observables 'ZIII', 'IZII'
        z1_op = eval('Z'+'^I'*(n_qubits-1))
        z2_op = eval('I'+'^Z'+'^I'*(n_qubits-2))
        
        pred_list = []
        for x in X_dat:
            # initial state encoding x
            self.circ_fm(x)
            state_init = StateFn(self.qc_fm)
        
            z1 = (~state_init @ ~qc_w_parameter @ z1_op @ qc_w_parameter\
                  @ state_init).eval().real
            z2 = (~state_init @ ~qc_w_parameter @ z2_op @ qc_w_parameter \
                  @ state_init).eval().real
            pred_list.append((z1, z2))
            
        return np.array(pred_list)
    
    @staticmethod
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
    
    
    def loss_accu(self, X_dat, Y_dat, theta):
        """
        
    
        Parameters
        ----------
        X : 2d ndarray, float, (rows, 2)
        Y : 1d ndarray, integer, (rows,)
        theta : 1d ndarray, float, (3*n_qubits,)
        qc : ComposedOp instance with both feature and variational ansatz
    
        Returns
        -------
        l: float, averaged cross entropy loss, for each data, loss entropy is 
            defined as -ylog(softmax(z1))-(1-y)log(softmax(z2))    
    
        """   
            
            
        l = 0
        accu_count  = 0
        
        pred_list = self.predictions(X_dat, theta)
        for (z1, z2), y in zip(pred_list, Y_dat):
            zz1, zz2 = self.softmax((z1, z2))
            l += -(1-y)*np.log(zz1)-y*np.log(zz2)  
        for (z1, z2), y in zip(pred_list, Y_dat):
            if (z2-z1)*(y-0.5)> 0:
                accu_count +=1
        return l/len(Y_dat), accu_count/len(Y_dat)
        
    
    def gradient(self, X_dat, Y_dat, theta):
        """
        Parameters
        ----------
        X_dat: 2d ndarray, float, input features, (rows, 2)
        Y_dat: 1d ndarray, int, true lables, (rows, )
        theta: 1d ndarray, float, to be optimized parameter in unitary circuits, (3*n_qubits*D)
        loss_track: list, float, record of loss function 
        accu_track: list, float, record of accuracy
        
        Returns
        -------
        grad: 1d ndarray, float, dloss/dtheta_i, (3*n_qubits*D,)
            -y (1-softmax(x_1))\partial x_1/\partial theta_i -(1-y)(1-softmax(x_2))\
                \partial x_2/\partial theta_i
    
        """  
    # =============================================================================
    #     
    #     loss_track.append(loss(X_dat, Y_dat, theta))
    #     accu_track.append(accuracy(X_dat, Y_dat, theta))
    #     plt.plot(range(len(loss_track)), loss_track)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Cross Entropy Loss Function')
    #     plt.show()
    #     print(f'accuarcy so far is {accu_track}')
    # =============================================================================
                 
        grad_list = []
        dtheta = np.pi/2* np.eye(len(theta))
        pred_list_theta = self.predictions(X_dat, theta)
        pred_list_theta_softmax = np.array(list(map(self.softmax, pred_list_theta)))
        for dt in dtheta:
            pred_list_plus = self.predictions(X_dat, theta+dt)
            pred_list_minus = self.predictions(X_dat, theta-dt)
            pred_list_dt = (pred_list_plus-pred_list_minus)*0.5
            grad_list_theta = -Y_dat * (1-pred_list_theta_softmax[:,1])*\
                pred_list_dt[:,1] -(1-Y_dat)*(1-pred_list_theta_softmax[:,0])\
                    *pred_list_dt[:,0]
            grad_list.append(np.sum(grad_list_theta, axis = 0)/len(Y_dat))
        return np.array(grad_list)
            
            
    
        
        
    def accuracy(self, X_dat, Y_dat, theta):
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
        pred = self.predictions(X_dat, theta, self.qc)
        accu_count = 0
        for (z1, z2), y in zip(pred, Y_dat):
            if (z2-z1)*(y-0.5)> 0:
                accu_count +=1
        return accu_count/len(Y_dat)
        
    


def adam(X_dat, Y_dat, qcl, minibatch_size = None, maxiter = 50):
    #loss_track keeps record of the loss function during the training process
    loss_track = []
    accu_track = []
    n = len(Y_dat)
    if not minibatch_size:
        minibatch_size = len(Y_dat)
    X_batched = np.array([X_dat[i:i+minibatch_size] for i in range(0, n, minibatch_size)])
    Y_batched = np.array([Y_dat[i:i+minibatch_size] for i in range(0, n, minibatch_size)])
    theta = np.random.uniform(size = 3*n_qubits*D)
    tol = 1e-3
    lr = 2* 1e-2 
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 1e-8
    t = 0
    m = np.zeros(3*n_qubits*D)
    v = np.zeros(3*n_qubits*D)
    for i in range(maxiter):
        for j in range(len(X_batched)):
            t += 1
            X_minibatch = X_batched[j]
            Y_minibatch = Y_batched[j]
            derivative = qcl.gradient(X_minibatch, Y_minibatch, theta)
            m = beta_1 * m + (1 - beta_1) * derivative    
            v = beta_2 * v + (1 - beta_2) * derivative * derivative
            lr_eff = lr * (np.sqrt(1- beta_2 ** t)) / (1- beta_1**t)
            theta = theta - lr_eff * m / (np.sqrt(v)+epsilon)
            result = qcl.loss_accu(X_dat, Y_dat, theta)
            loss_track.append(result[0])
            accu_track.append(result[1])
            if t>1 and np.abs(loss_track[-1]-loss_track[-2]) < tol:
                break
        fig, ax = plt.subplots(1,2)
        ax[0].plot(range(len(loss_track)), loss_track)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Cross Entropy Loss Function')
        ax[1].plot(range(len(accu_track)), accu_track)
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Accuracy')
        plt.show() 
    return theta   
    
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
    
if __name__ == '__main__':
    # circuit parameters
    n_qubits = 4
    T = 10 
    D = 2
    
    qcl = QuantumCircuitLearning(n_qubits, T, D)

    
    # Get training data from file if file exists already, otherwise generate with get_data
    n_samples = 10
    file_name = f'circle_data_n_{n_samples}'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            X_dat, Y_dat = pickle.load(f)
    else:
        X_dat, Y_dat = get_data(n_samples = n_samples, plot = True) 
        with open(file_name,'wb') as f:
            pickle.dump((X_dat, Y_dat), f)
    
    #theta = np.random.uniform(size = 3*n_qubits)
    #g = gradient(X_dat, Y_dat, theta)
    
    # Training with adam
    t1 = time()
    minibatch_size = 10
    maxiter = 20
    print(f'Training started for n_qubits = {n_qubits}, n_sample = {len(Y_dat)}, minibatch size = {minibatch_size}, maxiter = {maxiter}')
    theta = adam(X_dat, Y_dat, qcl,minibatch_size, maxiter)
    t2 = time()
    print(f'training takes {t2-t1} time')
# =============================================================================
#     accu_train = qcl.accuracy(X_dat, Y_dat, theta)
#     print(f'accuracy for training data is {accu_train}')
# =============================================================================
    
    
    
    #Testing  
    X_test, Y_test = get_data(n_samples = 20, plot = True) 
    loss_test, accu_test = qcl.loss_accu(X_test, Y_test, theta)
    print(f'loss function on test data is {loss_test}, accuracy on test data is {accu_test}')
    
    #Plot prediction
    X_test_plot = np.array([[(i,j) for i in np.arange(-1, 1, 0.2)] for j in np.arange(-1,1,0.2)])
    Y_test_plot = np.zeros(X_test_plot.shape[:2])
    for j in range(10):
        pred = qcl.predictions(X_test_plot[:,j].reshape(-1,2), theta)
        Y_test_plot[:,j] = np.array(list(map(qcl.softmax, pred)))[:,0]
    fig, (ax1, ax2) = plt.subplots(1,2)
    img1 = ax1.imshow(Y_test_plot, vmin = 0, vmax = 1)
    plt.colorbar(img1, ax = ax1)
    img2 = ax2.contourf(Y_test_plot, levels = [0,0.5,1], colors = ['red', 'purple'], alpha = 0.8)
    plt.colorbar(img2, ax = ax2)
    plt.axis('square')
    plt.show()