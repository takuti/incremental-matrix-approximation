# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

PATH_USPS='../data/usps/zip.train'

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
  data_mat = data_mat.T

  # k is truncate rank
  # optimal k will be around 10 because hand-written data only has 10 different digits (0,1,2,...,9)
  k = 10
  U, s, V = ln.svd(data_mat, full_matrices=False)
  Uk = U[:, :k]
  sk = s[:k]
  Vk = V[:k, :]
  data_mat_k = np.dot(np.dot(Uk, np.diag(sk)), Vk)

  for ell in [4, 8, 16, 32, 64, 128, 256]:
    U, s, V = frequent_directions(data_mat, ell)
    B = np.dot(np.dot(U, np.diag(s)), V)
    projected_mat = np.dot(np.dot(U[:, :k], U[:, :k].T), data_mat)

    cov = ln.norm(np.dot(data_mat, data_mat.T) - np.dot(B, B.T), ord=2)
    cov_err = cov / squaredFrobeniusNorm(data_mat)

    proj_err = squaredFrobeniusNorm(data_mat - projected_mat) / squaredFrobeniusNorm(data_mat - data_mat_k)

    print ell, cov_err, proj_err

def squaredFrobeniusNorm(mat_a):
  """Compute the squared Frobenius norm of given matrix
  """
  return ln.norm(mat_a, ord = 'fro') ** 2

if __name__ == '__main__':
  main()
