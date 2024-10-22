"""MIT License

Copyright (c) 2020 y-z-zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.linalg import block_diag
from scipy.stats import ortho_group

############### PURPOSE #################
# This code finds the finest simultaneous block diagonalization (SBD) of a set of symmetric matrices A.
# It also works for non-symmetric matrices, in such cases the SBD is the finest in the sense of matrix *-algebra.

############### USEAGE #################
# [P,BlockSizes] = sbd(A,threshold)
# A --- list containing the set of matrices to be simultaneously block diagonalized
# threshold --- real number, matrix entries below threshold are considered zero
# P --- orthogonal transformation matrix that performs SBD on A
# BlockSizes --- array listing the size of each common block

############### REFERENCE #################
# Y. Zhang, V. Latora, and A. E. Motter, "Unified Treatment of Dynamical Processes on Generalized Networks: Higher-order, Multilayer, and Temporal Interactions"

def sbd(A,threshold):
	n = len(A[0])	# size of the matrices to be simultaneously block diagonalized
	m = len(A)		# number of matrices to be simultaneously block diagonalized
	BlockSizes = []	# initialize the array that lists the size of each common block

    # B is a random self-adjoint matrix generated by matrices from A (and their conjugate transposes)
	B = np.zeros((n,n))
	for p in range(m):
		B = B + np.random.normal()*(A[p]+A[p].transpose())

    # find the eigenvalues and eigenvectors of B
	D, V = la.eigh(B)

    # C is a matrix used to sort the column vectors of V (i.e., the base vectors)
    # such that the base vectors corresponding to the same common block are next to each other
	C = np.zeros((n,n))
	for p in range(m):
		C = C + np.random.normal()*(A[p]+A[p].transpose())
	C = V.transpose()@C@V
	#for p in range(m):
	#	C = C + np.random.normal()*V.transpose()@A[p]@V

    # arrays used to track which base vectors have been sorted
	remaining_basis = list(range(n))
	sorted_basis = []

    # the sorting process: find C_ij's that are nonzero and group the base vectors v_i and v_j together
	while len(remaining_basis) > 0:
		current_block = [remaining_basis[0]]
		current_block_size = 1
		if len(remaining_basis) > 1:
			for idx in remaining_basis[1:]:
				if np.abs(C[remaining_basis[0],idx]) > threshold:
					current_block.append(idx)
					current_block_size = current_block_size + 1

		for idx in current_block:
			sorted_basis.append(idx)
			remaining_basis.remove(idx)

		# do the following in case there are zero entries inside the common block
		current_block_extra = []
		if len(remaining_basis) > 0:
			for idx in remaining_basis:
				for ind in current_block:
					if np.abs(C[ind,idx]) > threshold:
						current_block_extra.append(idx)
						current_block_size = current_block_size + 1
						break

		for idx in current_block_extra:
			sorted_basis.append(idx)
			remaining_basis.remove(idx)

		BlockSizes.append(current_block_size)

    # the sorted base vectors give the final orthogonal/unitary transformation matrix that performs SBD on A
	P = V[:,sorted_basis]

	return P, BlockSizes