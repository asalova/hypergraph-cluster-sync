import numpy as np
import numpy_indexed as npi
import sys
import itertools
from itertools import permutations 
from collections import Counter
import scipy.io as sio
import sbd
import pickle
import copy
from itertools import product
import cluster_calcs

class ClusterSync():
    '''
    Define the cluster synchronization pattern and check if it is admissible for a given coupling topology. 
    Find the matrix that block diaagonalizes the Jacobian by simultaneously block diagonalizing a set of 
    matrices corresponding to a given pattern of synchronization.
    '''
    def __init__(self, edges, clusters):
        '''
        Initialize the state
        
        Parameters
        ----------
        edges: dictionary
            list of edges for each order (dict. keys), with node labels starting at 1
        clusters: list
            list of lists, where each sub-list is a list of nodes belonging to the same cluster
        orders: list
            each element is a string corresponding to the order of edges present in the hypergraph
        '''
        self.orders = edges.keys()
        self.edge_types = {}
        self.nclusters = {}
        for order in self.orders:
            self.edge_types[order] = edges[order].keys()
        flat_clusters = [item for sublist in clusters for item in sublist]
        self.size = max(flat_clusters)
        self.edges = {}
        self.clusters = {}
        self.edges = {}
        for order in self.orders:
            self.edges[order] = {}
            for edge_type in self.edge_types[order]:
                self.edges[order][edge_type] = npi.unique(np.sort(np.array(edges[order][edge_type]))) - 1
        self.clusters['1'] = []  
        for cluster in clusters:
            cl = [c-1 for c in cluster]
            self.clusters['1']+= [cl]
        self.nclusters['1'] = len(clusters)
        self.incidence_from_edges() # getting the incidence matrices from edges
        self.cluster_edges() # getting the edge clusters based on (node) clusters
        self.incidence_effective() # effective incidence matrix used in cluster (quotient) dynamics 
            
    def incidence_from_edges(self):
        '''
        Obtain the incidence matrix
        '''
        self.I = {}
        for order in self.orders:
            self.I[order] = {}
            for edge_type in self.edge_types[order]:
                self.I[order][edge_type] = np.zeros((self.size, len(self.edges[order][edge_type])), dtype=float)
                for ind, edge in enumerate(self.edges[order][edge_type]):
                    for node in edge:
                        self.I[order][edge_type][node, ind]+= 1 
                
    def cluster_edges(self):
        '''
        Cluster the edges based on the cluster assignment of their nodes
        '''
        self.unique_clusters = {}
        self.edge_clusters = {}
        for order in self.orders:
            self.nclusters[order] = {}
            self.clusters[order] = {}
            self.unique_clusters[order] = {}
            self.edge_clusters[order] = {}
            for edge_type in self.edge_types[order]:
                self.clusters[order][edge_type] = np.zeros(np.shape(self.edges[order][edge_type]), dtype=int)
                for cluster_ind, cluster in enumerate(self.clusters['1']):
                    for node in cluster:
                        self.clusters[order][edge_type][np.where(self.edges[order][edge_type]==node)] = cluster_ind
                self.clusters[order][edge_type] = np.sort(self.clusters[order][edge_type], axis=1) 
                self.unique_clusters[order][edge_type] = npi.unique(self.clusters[order][edge_type])
                self.edge_clusters[order][edge_type] = np.zeros(len(self.edges[order][edge_type]), dtype=int) 
                for ind_unique, edge_unique in enumerate(self.unique_clusters[order][edge_type]):
                    for ind, edge in enumerate(self.clusters[order][edge_type]):
                        if set(edge)==set(edge_unique):
                            self.edge_clusters[order][edge_type][ind] = ind_unique
                self.nclusters[order][edge_type] = np.max(self.edge_clusters[order][edge_type]) + 1
        
    def incidence_effective(self):
        '''
        Check if the cluster assignment is valid. If it is, obtain the effective incidence 
        matrix of the quotient network.
        '''
        self.I_P = {}
        self.I_clust = {} 
        self.I_eff = {}  
        self.unique_clusters_L = {} 
        self.I_clust_L = {}
        for order in self.orders: 
            self.I_P[order] = {}
            self.I_clust[order] = {}
            self.unique_clusters_L[order] = {}
            self.I_clust_L[order] = {}
            self.I_eff[order] = {}
            
            for edge_type in self.edge_types[order]:

                self.I_P[order][edge_type] = np.zeros((len(self.clusters[order][edge_type]),\
                                            len(self.unique_clusters[order][edge_type])),\
                                            dtype=int)
                for ind_edge, edge in enumerate(self.unique_clusters[order][edge_type]):
                    self.I_P[order][edge_type][:, ind_edge] = (self.edge_clusters[order][edge_type]==ind_edge)
                self.I_clust[order][edge_type] = self.I[order][edge_type] @ self.I_P[order][edge_type]
                        
                self.unique_clusters_L[order][edge_type] = np.empty((0,int(order)), dtype=int)
                self.I_clust_L[order][edge_type] = np.empty((0,self.size), dtype=int)
                for ind_edge, edge in enumerate(self.unique_clusters[order][edge_type]):
                    if len(set(edge))!= 1:
                        self.unique_clusters_L[order][edge_type] = np.vstack((self.unique_clusters_L[order][edge_type],\
                                                                   edge))
                        self.I_clust_L[order][edge_type] = np.vstack((self.I_clust_L[order][edge_type],\
                                                           self.I_clust[order][edge_type][:, ind_edge].T))
                self.I_clust_L[order][edge_type] = np.transpose(self.I_clust_L[order][edge_type])
                
                self.I_eff[order][edge_type] = np.zeros((len(self.clusters['1']),\
                                              len(self.unique_clusters_L[order][edge_type])),\
                                              dtype=int)
                for i, cluster in enumerate(self.clusters['1']):
                    test_mat = self.I_clust_L[order][edge_type][cluster]
                    if not (test_mat == test_mat[0]).all():
                        sys.exit('Not a valid cluster assignment')
                    else:
                        self.I_eff[order][edge_type][i, :] = test_mat[0]
            
    def SBD(self, err):
        '''
        Create a list of matrices that need to be block diagonalized. Perform the 
        simultaneous block diagonalization.
        
        Parameters
        ----------
        err: float
            error bound that will be used in SBD code
        '''
        self.L_BD = {}
        self.adjacency_pattern()
        self.SBD_input = list(self.E)
        for order in self.orders:
            for edge_type in self.edge_types[order]:
                self.SBD_input+= list(self.L_patterns[order][edge_type])
        self.U_SBD, self.block_size_SBD = sbd.sbd(self.SBD_input, err)
        self.n_transverse_blocks = len(self.block_size_SBD) - 1
        self.block_inds = np.zeros(len(self.block_size_SBD) + 1, dtype=int)
        for i, blocks in enumerate(self.block_size_SBD):
            self.block_inds[i+1] = np.sum(self.block_size_SBD[:i+1])
        self._remove_parallel_perturbations()
        
        self.Jacobian_input = {}
        self.unique_clusters_dyn = {}
        self.unique_clusters_dyn_perm = {}
        self.E_BD = self._BD(self.E)
        for order in self.orders:
            self.Jacobian_input[order] = {}
            self.unique_clusters_dyn_perm[order] = {}
            self.unique_clusters_dyn[order] = {}
            for edge_type in self.edge_types[order]:
                self.Jacobian_input[order][edge_type] = {}
                self.unique_clusters_dyn[order][edge_type] = []
                self.unique_clusters_dyn_perm[order][edge_type] = []
                L = self._BD(self.L_patterns[order][edge_type])
                u = 0
                upd = 0
                for i, pattern in enumerate(self.unique_clusters[order][edge_type]):
                    unique_nodes, inds = np.unique(pattern, return_index = True)
                    if upd==1:
                        u+= 1
                    for ind in inds:
                        p = copy.copy(pattern)
                        p[[0, ind]] = p[[ind, 0]]
                        L_BD = []
                        for k in range(self.n_transverse_blocks):
                            L_BD.append(self.E_BD[k][p[0]] @ L[k][i])
                        self.Jacobian_input[order][edge_type][tuple(p)] = L_BD
                        if len(np.unique(p))>1:
                            self.unique_clusters_dyn[order][edge_type]+= [p] 
                            self.unique_clusters_dyn_perm[order][edge_type]+= [u]
                            upd = 1
                        else:
                            upd = 0
                    
    def adjacency_pattern(self):
        '''
        Create a list of matrices that need to be block diagonalized.
        '''
        self.A_patterns = {}
        self.L_patterns = {}
        self.E = np.zeros((self.nclusters['1'], self.size, self.size))
        for i, cluster in enumerate(self.clusters['1']):
            v = np.zeros(self.size, dtype=int)
            v[cluster] = 1 
            self.E[i, :, :] = np.diag(v)
        for order in self.orders:  
            self.A_patterns[order] = {}
            self.L_patterns[order] = {}
            for edge_type in self.edge_types[order]:
                self.A_patterns[order][edge_type] = np.zeros((self.nclusters[order][edge_type],) + (self.size,)*2)
                self.L_patterns[order][edge_type] = np.zeros((self.nclusters[order][edge_type],) + (self.size,)*2)
                for i in range(self.nclusters[order][edge_type]):
                    I = self.I[order][edge_type][:, self.edge_clusters[order][edge_type]==i]
                    A = I @ I.T
                    diag = np.diag(np.diagonal(A))
                    self.A_patterns[order][edge_type][i,:,:] = A - diag
                    self.L_patterns[order][edge_type][i,:,:] = A - int(order) * diag
                
    def _remove_parallel_perturbations(self):
        errs = np.zeros(self.size)
        for i in range(self.size):
            for cluster in self.clusters['1']:
                vect_cluster = self.U_SBD[cluster, i] 
                errs[i]+= np.sum(np.abs(vect_cluster\
                                      - np.mean(vect_cluster, axis=0)))
        inds = np.argsort(errs)[:self.nclusters['1']]  
        self.U_SBD_transverse = np.delete(self.U_SBD, inds, axis=1)
        ind = np.min(inds)
        ind = np.where(self.block_inds == ind)[0]
        self.block_size_SBD_transverse = np.delete(self.block_size_SBD, ind)
        self.block_inds_transverse = np.zeros(len(self.block_size_SBD), dtype=int)
        for i in range(len(self.block_size_SBD_transverse)):
            self.block_inds_transverse[i+1] = np.sum(self.block_size_SBD_transverse[:i+1])
                
    def _BD(self, mats):
        M = [None] * len(mats)
        for i, mat in enumerate(mats):
            M[i] = self.U_SBD_transverse.T @ mat @ self.U_SBD_transverse
        M_BD = [None] * self.n_transverse_blocks
        for i in range(self.n_transverse_blocks):
            bs = self.block_size_SBD_transverse[i]
            block = np.zeros((len(mats), bs, bs))
            for j, m in enumerate(M):
                block[j, :, :]= m[self.block_inds_transverse[i]:self.block_inds_transverse[i+1],
                                  self.block_inds_transverse[i]:self.block_inds_transverse[i+1]]
            M_BD[i] = block 
        return M_BD