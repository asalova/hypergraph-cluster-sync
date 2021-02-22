import numpy as np
import numpy_indexed as npi
import sys
import itertools
from itertools import permutations 
from itertools import product
from collections import Counter
import scipy.io as sio
import sbd
import pickle
import copy
import cluster_calcs

class Dynamics():
    '''
    Evolve the dynamics together with the pertrubations to obtain the maximum transverse
    Lyapunov exponent.
    '''
    def __init__(self, cluster_sync, dynamics_terms, jacobian_terms, beta, sigma):
        '''
        Initialize the state.
        
        Parameters
        ----------
        cluster_sync: ClusterSync instance
            information about the cluster synchronization pattern and the SBD transformed 
            Jacobian elements
        dynamics_terms: dictionary of functions
            evolution of clusters
        jacobian_terms: dictionary of functions
            evolution of the Jacobian elements
        beta: float
            self evolution parameter
        sigma: dictionary
            each dictionary element corresponds to a set of coupling strengths for a given 
            edge order
        '''
        self.cs = cluster_sync
        self.beta = beta
        self.sigma = sigma
        self.dynamics_terms = dynamics_terms
        self.jacobian_terms = jacobian_terms
                
    def dynamics_step(self, IC):
        '''
        Evolve the quotient network dynamics by one timestep.
        
        Parameters
        ----------
        IC: numpy array
            the state at the previous time step
        '''
        next_step = self.beta * self.dynamics_terms['1'](IC)
        for order in self.cs.orders:
            for edge_type in self.cs.edge_types[order]:
                for i, edge in enumerate(self.cs.unique_clusters_dyn[order][edge_type]):
                    next_step[edge[0]]+=  self.cs.I_eff[order][edge_type][edge[0], self.cs.unique_clusters_dyn_perm[order][edge_type][i]]\
                                          * self.sigma[order][edge_type]\
                                          * self.dynamics_terms[order][edge_type](IC[edge])
        return next_step
    
    def perturbation_step(self, IC_d, IC_p):
        '''
        Evolve the perturbation dynamics by one timestep.
        
        Parameters
        ----------
        IC_d: numpy array
            the state of the quotient dynamics at the previous time step
        IC_p: numpy array
            the state of the perturbation dynamics at the previous time step
        '''
        next_step = np.zeros(len(IC_p))
        for i in range(len(self.cs.block_inds_transverse) -1):
            i_low = self.cs.block_inds_transverse[i]
            i_high = self.cs.block_inds_transverse[i+1]
            ns = np.zeros((self.cs.block_size_SBD_transverse[i], 
                           self.cs.block_size_SBD_transverse[i]))
            for j, E_BD in enumerate(self.cs.E_BD[i]):
                ns+= self.beta * E_BD\
                     * self.jacobian_terms['1'](IC_d[j])
            for order in self.cs.orders:
                for edge_type in self.cs.edge_types[order]:
                    for key in self.cs.Jacobian_input[order][edge_type]:
                        L_BD = self.cs.Jacobian_input[order][edge_type][key][i]
                        ns+= self.sigma[order][edge_type] * L_BD * self.jacobian_terms[order][edge_type](IC_d[list(key)])  
                next_step[i_low:i_high] = ns @ IC_p[i_low:i_high]
        le = np.log(np.linalg.norm(next_step))
        next_step = next_step/np.linalg.norm(next_step)
        
        return next_step, le
    
    def extended_dynamics_evolution(self, T):
        '''
        Evolve the extended system for a specified number of time steps
        
        Parameters
        ----------
        T: int
            number of times the dynamics is evolved
        '''
        self.T = T
        IC_d = np.random.rand(self.cs.nclusters['1']) * 2 * np.pi
        IC_p = (np.random.rand(self.cs.size - self.cs.nclusters['1']) - 0.5) * 0.1
        IC_p = IC_p/np.linalg.norm(IC_p)
        dynamics_quotient = np.zeros((T, self.cs.nclusters['1']))
        dynamics_quotient[0, :] = IC_d
        dynamics_perturbations = np.zeros((T, self.cs.size - self.cs.nclusters['1']))
        dynamics_perturbations[0, :] = IC_p
        norms = np.zeros((T-1,))
        for i in range(T-1):
            dynamics_quotient[i+1, :] = self.dynamics_step(dynamics_quotient[i, :])           
            dynamics_perturbations[i+1, :], norms[i] = self.perturbation_step(dynamics_quotient[i, :], 
                                                                              dynamics_perturbations[i, :])
        return dynamics_quotient, dynamics_perturbations, norms
