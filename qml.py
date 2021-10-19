# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:45:54 2021

@author: Huijie Guan

Pick More Daisies
"""
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit, execute
from qiskit import Aer
from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp

class QML:
    def __init__(self, n_qubit, encoding = ZZFeatureMap, variation = RealAmplitudes, reps_fm =1, reps_v = 1, backend = 'aer_simulator'):
        self.n_qubit = n_qubit
        self.reps_v = reps_v
        self.fm = encoding(self.n_qubit, reps = reps_fm)
        self.variation = variation(self.n_qubit, reps = reps_v)
        self.circ = QuantumCircuit(self.n_qubit)
        self.circ = self.circ.compose(self.fm).compose(self.variation)
        self.circ.measure_all()
        self.backend = Aer.get_backend(backend)
        self.shots = 1000
        self.n_var = self.n_qubit * (reps_v+1)
        
        
    def circuit_output(self,theta, x):
        """
        params: theta: 1D array of [theta[0], theta[1]..]
                x: 1D array of [x[0], x[1], ...]
        return: result from executing for a particular set of theta and X
        """
        
        parameters = np.hstack((theta,x))
        circuit = self.circ.assign_parameters(parameters)
        job = execute(circuit, self.backend, shots = self.shots)
        stat = job.result().get_counts()
        return stat
        
    def get_prob(self, stat, y):
        """
        :params: stat, a dictionary of the form {'000': 2, '001':3,...}
                 y: integer, label
        :return: probability(x,y, theta)
        """
        p = 0
        for key, value in stat.items():
            if key.count('1')%2 ==y:
                p += value
        return p/self.shots
        
    def get_diff(self, theta, x, y):
        """
        :params: theta:  1D array of [theta[0], theta[1]..]
                 x: 1D array of [x[0], x[1], ...]
                 y: integer, label
        :return: 1D array of [dp/dtheta_1, dp/dtheta_2, ...]
        """
        dp = np.zeros(self.n_var)
        p = self.get_prob(self.circuit_output(theta, x), y)
        dtheta = np.zeros([self.n_var, self.n_var])
        for i,dta in enumerate(dtheta):
            p_prime = self.get_prob(self.circuit_output(theta+dta, x), y)
            dp[i] = p_prime
        return dp
    
    def fisher_information_matrix(self, theta, x, y):
        """
        :params: theta:  1D array of [theta[0], theta[1]..]
                 x: 1D array of [x[0], x[1], ...]
                 y: integer, label   
        :return: fi_matrix for a particular x, theta
        
        """
        fi_vec = self.get_diff(theta, x, y)
        return np.outer(fi_vec, fi_vec)    
def get_data(rows = None):
    file = "C:\\Users\\86153\\iris_csv.csv"
    data = pd.read_csv(file)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    data_filter = (y != 'Iris-virginica')
    X_f = X[data_filter]
    y = y[data_filter]
    if rows:
        X = X.sample(rows)
        y = y.sample(rows)
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    return np.array(X), np.array(y)

X, y = get_data(1)
n_qubit = len(X[0])
reps_v =1
n_v = n_qubit *(reps_v+1)
theta = np.random.rand(n_v)
f = QML(n_qubit, reps_v = reps_v)
f.fisher_information_matrix(theta, x=X[0], y=y[0])
            
    