# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

def frequent_directions(A, ell):
  """A matrix "A" should be 256x7291
  """

  m = 256
  n = 7291

  if A.shape[0] != m or A.shape[1] != n: raise ValueError('Error: incorrect matrix size')

  start = time.clock()

  B = np.hstack((A[:, :(ell-1)], np.zeros((m, 1))))

  for i in range(ell-1, n):

    # new matrix is just a single vector (i-th column of A)
    B[:, ell-1] = A[:, i]
    U, s, V = ln.svd(B, full_matrices=False)

    delta = s[-1] ** 2 # squared smallest singular value

    B = np.dot(U, np.diag(np.sqrt(abs(s ** 2 - delta))))

  U, s, V = ln.svd(B, full_matrices=False)

  elapsed_time = time.clock() - start
  print 'time:', elapsed_time

  return U, s, V
