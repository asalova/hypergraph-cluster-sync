import numpy as np
import numpy_indexed as npi
import sys
import itertools
from itertools import permutations 
from itertools import product
from collections import Counter
import scipy.io as sio
import pickle
import copy
import cluster_calcs

def deviation_test(T, s_3, sigmas_2, n_sig_2, beta, IC):
    '''
    Calculate the deviations from cluster synchronization as noise is added.
    
    Parameters
    ----------
    T: int
        number of time steps
    s_3: int
        index of the triadic coupling parameter
    sigmas_2: array 
        all the dyadic coupling parameter values
    n_sig_2: int
        number of dyadic parameters
    beta: float
        beta parameter
    IC: numpy array 
        initial conditions for all sigmas and clusters, n_sig_2 by the number of clusters
        
    Returns
    -------
    errors: numpy array
        errors for all the parameter values
    '''
    errors = np.zeros(n_sig_2)
    for i, s_2 in enumerate(sigmas_2):
        IC_s2 = IC[i, :] 
        dynamics = full_dyn(s_2, s_3, beta, T, IC_s2)
        dyn = dynamics[:, -12000:]
        err = np.mean(np.std(dyn[:7, :], axis=0))
        err+= np.mean(np.std(dyn[7:, :], axis=0))
        errors[i] = err
    return errors

def full_dyn(s_2, s_3, beta, T, IC_s2):
    '''
    Evolve the dynamics.
    
    Parameters:
    s_2: float
        dyadic coupling
    s_3: float
        triadic coupling
    beta: float
        beta parameter
    T: int
        trajectory length
    IC_s_2: numpy array
        initial condition for each cluster
    '''
    dynamics = np.zeros((14, T))
    dynamics[:, 0] = [IC_s2[0]] * 7 + [IC_s2[1]] * 7 + (np.random.normal(size=14))*.01
    for i in range(T-1):
        next_step = np.zeros(14)
        prev_step = dynamics[:, i] 
        for j in range(7):
            next_step[j] = beta * (np.sin(prev_step[j] + np.pi/4))**2 
            next_step[j+7] = beta * (np.sin(prev_step[j+7] + np.pi/4))**2
            
            next_step[j]+= s_2 * ((np.sin(prev_step[(j+1)%7] + np.pi/4))**2\
                                  - (np.sin(prev_step[j] + np.pi/4))**2)
            next_step[j+7]+= s_2 * ((np.sin(prev_step[(j+1)%7 + 7] + np.pi/4))**2\
                                    - (np.sin(prev_step[j+7] + np.pi/4))**2)
            next_step[j]+= s_2 * ((np.sin(prev_step[(j-1)%7] + np.pi/4))**2\
                                  - (np.sin(prev_step[j] + np.pi/4))**2)
            next_step[j+7]+= s_2 * ((np.sin(prev_step[(j-1)%7 + 7] + np.pi/4))**2\
                                    - (np.sin(prev_step[j+7] + np.pi/4))**2)
            
            next_step[j]+= s_2 * ((np.sin(prev_step[j+7] + np.pi/4))**2\
                                  - (np.sin(prev_step[j] + np.pi/4))**2)
            next_step[j+7]+= s_2 * ((np.sin(prev_step[j] + np.pi/4))**2\
                                    - (np.sin(prev_step[j+7] + np.pi/4))**2)

            next_step[j]+= s_2 * ((np.sin(prev_step[(j+2)%7] + np.pi/4))**2\
                                  - (np.sin(prev_step[j] + np.pi/4))**2)
            next_step[j+7]+= s_2 * ((np.sin(prev_step[(j+2)%7 + 7] + np.pi/4))**2\
                                    - (np.sin(prev_step[j+7] + np.pi/4))**2)
            next_step[j]+= s_2 * ((np.sin(prev_step[(j-2)%7] + np.pi/4))**2\
                                  - (np.sin(prev_step[j] + np.pi/4))**2)
            next_step[j+7]+= s_2 * ((np.sin(prev_step[(j-2)%7 + 7] + np.pi/4))**2\
                                    - (np.sin(prev_step[j+7] + np.pi/4))**2)
            
            next_step[j]+= - s_3 * np.sin(prev_step[(j+1)%7] + prev_step[(j-1)%7]\
                                          - 2 * prev_step[j])
            next_step[j+7]+= - s_3 * np.sin(prev_step[(j+1)%7+7] + prev_step[(j-1)%7+7]\
                                            - 2 * prev_step[j+7])
        next_step+= (np.random.normal(size=14))*.01
            
        dynamics[:, i+1] = next_step 
    return dynamics