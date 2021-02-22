# hypergraph-cluster-sync
Admissibility and stability of cluster synchronization patterns on hypergraphs.

This repository is part of the manuscript: A Salova, RM D'Souza, _Cluster synchronization on hypergraphs_, arXiv preprint arXiv:2101.05464.

The repository includes:

1. 'example.ipynb'

   _Jupyter notebook file that provides an example of running the code to find analyze the admissibility and stability of a specific cluster 
   synchronization pattern on a hypergraph._
   
2. 'sbd.py'

   _Copy of the implementation of the simultaneous block diagonalization algorithm from https://github.com/y-z-zhang/SBD._
   
3. 'cluster_calcs.py'

   _Tests the admissibility of a specific cluster synchronization pattern based on the hypergraph structure. Accepts hypergraphs with arbitrary 
   edge orders and number of edge types. Provides a set of matrices that need to be simultaneously block diagonalized, as well as their block
   diagonalization using_ 'sbd.py'.
   
4. 'dynamics.py'

   _Estimating the Maximum Lyapunov Exponent for cluster synchronization patterns for a given parameter regime. Requires an input of dynamical 
   terms (node and coupling dynamics), as well as their partial derivatives._
   
5. 'mle.p'

    MLE estimates from the example in the manuscript.
