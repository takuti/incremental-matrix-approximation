# coding: utf-8

import numpy as np
import numpy.linalg as ln

from brute_force import brute_force
from incremental_svd import incremental_svd
from frequent_directions import frequent_directions

PATH_USPS='../data/usps/zip.train'

def squaredFrobeniusNorm(mat_a):
  """Compute the squared Frobenius norm of given matrix
  """
  return ln.norm(mat_a, ord = 'fro') ** 2

def exp_bf(A):
  """Brute force
  """

  U, s, V = brute_force(A)

def exp_isvd(A):
  """Incremental SVD
  """

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

def exp_fd(A):
  """Frequent Directions
  """

  # k is truncate rank
  # optimal k will be around 10 because hand-written data only has 10 different digits (0,1,2,...,9)
  k = 10
  U, s, V = ln.svd(A, full_matrices=False)
  Uk = U[:, :k]
  sk = s[:k]
  Vk = V[:k, :]
  Ak = np.dot(np.dot(Uk, np.diag(sk)), Vk)

  for ell in [4, 8, 16, 32, 64, 128, 256]:
    U, s, V = frequent_directions(A, ell)
    B = np.dot(np.dot(U, np.diag(s)), V)
    projected_mat = np.dot(np.dot(U[:, :k], U[:, :k].T), A)

    cov = ln.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)
    cov_err = cov / squaredFrobeniusNorm(A)

    proj_err = squaredFrobeniusNorm(A - projected_mat) / squaredFrobeniusNorm(A - Ak)

    print ell, cov_err, proj_err

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

  #exp_bf(A)
  exp_isvd(A)
  exp_fd(A)

if __name__ == '__main__':
  main()
