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
    def __init__(self, edges, clusters, orders=['2', '3']):
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
        self.orders = orders
        flat_clusters = [item for sublist in clusters for item in sublist]
        self.size = max(flat_clusters)
        self.edges = {}
        self.clusters = {}
        self.nclusters = {}
        for order in orders:
            self.edges[order] = npi.unique(np.sort(np.array(edges[order]))) - 1
        self.clusters['1'] = []  
        for cluster in clusters:
            cl = [c-1 for c in cluster]
            self.clusters['1']+= [cl]
        self.nclusters['1'] = len(clusters)
        self.adjacency_from_edges() # getting the adjacency tensors from edges
        self.incidence_from_edges() # getting the incidence matrix from edges
        self.cluster_edges() # getting the edge clusters based on (node) clusters
        self.incidence_effective() # effective incidence matrix used in cluster (quotient) dynamics
        self.incidence_to_adjacency_effective() # effective adjacency tensors
        
    def adjacency_from_edges(self):
        '''
        Obtain the adjacency tensors from edges
        '''
        self.A = {}
        for order in self.orders:
            self.A[order] = np.zeros((self.size,)*int(order), dtype=int)
            for edge in self.edges[order]:
                for p in permutations(edge):
                    self.A[order][p]+= 1 
            
    def incidence_from_edges(self):
        '''
        Obtain the incidence matrix
        '''
        self.I = {}
        for order in self.orders:
            self.I[order] = np.zeros((self.size, len(self.edges[order])), dtype=int)
            for ind, edge in enumerate(self.edges[order]):
                for node in edge:
                    self.I[order][node, ind] = 1
                
    def cluster_edges(self):
        '''
        Cluster the edges based on the cluster assignment of their nodes
        '''
        self.unique_clusters = {}
        self.edge_clusters = {}
        for order in self.orders:
            self.clusters[order] = np.zeros(np.shape(self.edges[order]), dtype=int)
            for cluster_ind, cluster in enumerate(self.clusters['1']):
                for node in cluster:
                    self.clusters[order][np.where(self.edges[order]==node)] = cluster_ind
            #self.clusters[order] = np.sort(self.clusters[order], axis=1) 
            self.unique_clusters[order] = npi.unique(self.clusters[order])
            self.edge_clusters[order] = np.zeros(len(self.edges[order]), dtype=int) 
            for ind_unique, edge_unique in enumerate(self.unique_clusters[order]):
                for ind, edge in enumerate(self.clusters[order]):
                    if set(edge)==set(edge_unique):
                        self.edge_clusters[order][ind] = ind_unique
            self.nclusters[order] = np.max(self.edge_clusters[order]) + 1
        
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
            self.I_P[order] = np.zeros((len(self.clusters[order]),\
                                        len(self.unique_clusters[order])),\
                                        dtype=int)
            for ind_edge, edge in enumerate(self.unique_clusters[order]):
                self.I_P[order][:, ind_edge] = (self.edge_clusters[order]==ind_edge)
            self.I_clust[order] = self.I[order] @ self.I_P[order]
                        
            self.unique_clusters_L[order] = np.empty((0,int(order)), dtype=int)
            self.I_clust_L[order] = np.empty((0,self.size), dtype=int)
            for ind_edge, edge in enumerate(self.unique_clusters[order]):
                if len(set(edge))!= 1:
                    self.unique_clusters_L[order] = np.vstack((self.unique_clusters_L[order],\
                                                               edge))
                    self.I_clust_L[order] = np.vstack((self.I_clust_L[order],\
                                                       self.I_clust[order][:, ind_edge].T))
            self.I_clust_L[order] = np.transpose(self.I_clust_L[order])
                
            self.I_eff[order] = np.zeros((len(self.clusters['1']),\
                                          len(self.unique_clusters_L[order])),\
                                          dtype=int)
            for i, cluster in enumerate(self.clusters['1']):
                test_mat = self.I_clust_L[order][cluster]
                if not (test_mat == test_mat[0]).all():
                    sys.exit('Not a valid cluster assignment')
                else:
                    self.I_eff[order][i, :] = test_mat[0]
                
    def incidence_to_adjacency_effective(self):
        '''
        Obtain the effective adjacency tensors from the incidence matrix.
        '''
        self.A_eff = {}
        self.L_eff = {}
        for order in self.orders: 
            self.A_eff[order] = np.zeros((self.nclusters['1'],)*int(order), dtype=int)
            for i in range(self.nclusters['1']):
                for j, edge in enumerate(self.unique_clusters_L[order]):
                    if self.I[order][i,j]==1:
                        for p in list(permutations(edge)):
                            self.A_eff[order][p] = 1
            self.L_eff[order] = self.A_eff[order] - np.diag(np.sum(self.A_eff[order], axis=1))

            
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
            self.SBD_input+= list(self.L_patterns[order])
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
            self.unique_clusters_dyn[order] = []
            self.unique_clusters_dyn_perm[order] = []
            L = self._BD(self.L_patterns[order])
            u = 0
            upd = 0
            for i, pattern in enumerate(self.unique_clusters[order]):
                unique_nodes, inds = np.unique(pattern, return_index = True)
                if upd==1:
                    u+= 1
                for ind in inds:
                    p = copy.copy(pattern)
                    p[[0, ind]] = p[[ind, 0]]
                    L_BD = []
                    for k in range(self.n_transverse_blocks):
                        L_BD.append(self.E_BD[k][p[0]] @ L[k][i])
                    self.Jacobian_input[order][tuple(p)] = L_BD
                    if len(np.unique(p))>1:
                        self.unique_clusters_dyn[order]+= [p] 
                        self.unique_clusters_dyn_perm[order]+= [u]
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
            self.A_patterns[order] = np.zeros((self.nclusters[order],) + (self.size,)*2)
            self.L_patterns[order] = np.zeros((self.nclusters[order],) + (self.size,)*2)
            for i in range(self.nclusters[order]):
                I = self.I[order][:, self.edge_clusters[order]==i]
                A = I @ I.T
                diag = np.diag(np.diagonal(A))
                self.A_patterns[order][i,:,:] = A - diag
                self.L_patterns[order][i,:,:] = A - int(order) * diag
                
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