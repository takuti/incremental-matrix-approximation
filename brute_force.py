# coding: utf-8

import numpy as np
import numpy.linalg as ln
import time
import sys

PATH_USPS='../data/usps/zip.train'

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

  U, s, V = brute_force(data_mat.T)

if __name__ == '__main__':
  main()
