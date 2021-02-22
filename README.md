# hypergraph-cluster-sync
**Admissibility and stability of cluster synchronization patterns on hypergraphs.**

This repository is part of the manuscript: A Salova, RM D'Souza, _Cluster synchronization on hypergraphs_, arXiv preprint arXiv:2101.05464.

The contents of the repository (Python 3 files and Jupyter Notebook) can be used to analyze the patterns of cluster synchronization on hypergraphs. 
With the input of an arbitrary hypergraph topology and a specific candidate cluster synchroinzation pattern, we test the admissibility of 
that pattern and provide a set of matrices sufficient to block diagoanalize the Jacobian. 
We then perform the block diagonalization using the file 'sbd.py' from https://github.com/y-z-zhang/SBD. 
With the input of node and coupling dynamics and parameter values, this block diagonalization is used to perform the linear stability calculation. 

This code is general in a sence that the input can include arbitrary edge coupling orders, as well as more than one type of 
edges.

**The repository includes:**

1. **'example.ipynb'**

   _Jupyter notebook file that provides an example of running the code to find analyze the admissibility and stability of a specific cluster 
   synchronization pattern on a hypergraph._
   
2. **'sbd.py'**

   _Copy of the implementation of the simultaneous block diagonalization algorithm from https://github.com/y-z-zhang/SBD._
   
3. **'cluster_calcs.py'**

   _Tests the admissibility of a specific cluster synchronization pattern based on the hypergraph structure. Accepts hypergraphs with arbitrary 
   edge orders and number of edge types. Provides a set of matrices that need to be simultaneously block diagonalized, as well as their block
   diagonalization using_ 'sbd.py'.
   
4. **'dynamics.py'**

   _Estimating the Maximum Lyapunov Exponent for cluster synchronization patterns for a given parameter regime. Requires an input of dynamical 
   terms (node and coupling dynamics), as well as their partial derivatives._
   
5. **'mle.p'**

    MLE estimates from the example in the manuscript.
    
**Required packages:**

- scipy
- copy
- itertools
- sys
- numpy_indexed