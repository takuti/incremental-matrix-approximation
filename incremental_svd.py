# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

PATH_USPS='../data/usps/zip.train'

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

def main():
  """Assume that columns of the original matrix come from data stream one-by-one
  """

  # load USPS hand-written data
  # original matrix will be 7291x256
  original_mat = np.loadtxt(PATH_USPS)
  labels = map(int, original_mat[:, 0].tolist())
  data_mat = original_mat[:, 1:]

  # mean shift to zero
  mean_sample = np.mean(data_mat, axis=0)
  data_mat -= mean_sample
  A = data_mat.T

  # k is truncate rank
  # optimal k will be around 10 because hand-written data only has 10 different digits (0,1,2,...,9)
  orig_U, orig_s, orig_V = ln.svd(A, full_matrices=False)

  U, s, V = incremental_svd(A, qr_flg=False)

  for k in [2, 4, 8, 16, 32, 64, 128, 256]:

    Uk = U[:, :k]
    sk = s[:k]
    Vk = V[:k, :]

    B = np.dot(np.dot(Uk, np.diag(sk)), Vk)

    cov = ln.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)
    cov_err = cov / squaredFrobeniusNorm(A)

    Ak = np.dot(np.dot(orig_U[:,:k], np.diag(orig_s[:k])), orig_V[:k,:])

    # for smallest singular value, squaredFrobeniusNorm will be very small
    # so, numerical error will be critical factor: A-UkUk^TA1 -> 1.69225950459e-21, A-Ak -> 5.50568459989e-22
    proj_err = squaredFrobeniusNorm(A - np.dot(np.dot(Uk, Uk.T), A)) / squaredFrobeniusNorm(A - Ak)

    print k, cov_err, proj_err

def squaredFrobeniusNorm(mat_a):
  """Compute the squared Frobenius norm of given matrix
  """
  return ln.norm(mat_a, ord = 'fro') ** 2

if __name__ == '__main__':
  main()
