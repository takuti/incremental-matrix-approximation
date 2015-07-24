# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

def incremental_svd(A, qr_flg=False):
  """A matrix "A" should be 256x7291
  """

  m = 256
  n = 7291

  n0 = 256

  if A.shape[0] != m or A.shape[1] != n: raise ValueError('Error: incorrect matrix size')

  start = time.clock()

  A0 = A[:, :n0]
  U, s, V = ln.svd(A0, full_matrices=False)

  # NOTE: s is a vector; np.diag(s) will produce a diagonal matrix
  for i in range(n0, n):

    # new matrix is just a single vector (i-th column of A)
    A1 = np.matrix(A[:, i]).T

    if qr_flg:
      J, K = ln.qr(A1 - np.dot(np.dot(U, U.T), A1))
      U_, s_, V_ = ln.svd(
          np.vstack((
            np.hstack((np.diag(s), np.dot(U.T, A1))),
            np.hstack((np.zeros((K.shape[0], s.shape[0])), K))
          )),
          full_matrices=False)

      # update the result of SVD
      U = np.dot(np.hstack((U, J)), U_)

    else:
      U_, s_, V_ = ln.svd(np.hstack((np.diag(s), np.dot(U.T, A1))), full_matrices=False)
      U = np.dot(U, U_)

    s = s_

    # NOTE: V from svd on NumPy is already transposed
    V = np.dot(V_,
          np.vstack((
            np.hstack((V, np.zeros((V.shape[0], i+1-V.shape[1])))),
            np.hstack((np.zeros((V_.shape[1]-V.shape[0], V.shape[1])), np.eye(V_.shape[1]-V.shape[0], i+1-V.shape[1])))
          ))
        )

    # for next computation, update A0
    A0 = np.hstack((A0, A1))

  elapsed_time = time.clock() - start
  print 'time:', elapsed_time

  return U, s, V
