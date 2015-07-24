# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

def brute_force(A):
  """A matrix "A" should be 256x7291
  """

  m = 256
  n = 7291

  if A.shape[0] != m or A.shape[1] != n: raise ValueError('Error: incorrect matrix size')

  start = time.clock()

  # NOTE: s is a vector; np.diag(s) will produce a diagonal matrix
  for i in range(256, n):
    U, s, V = ln.svd(A[:,:i+1], full_matrices=False)

  elapsed_time = time.clock() - start
  print elapsed_time

  return U, s, V
