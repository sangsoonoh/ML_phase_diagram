"""
Miscellenous utility functions
"""
import numpy as np
import h5py
import rich

def trunc_svd(mat, r_threshold = 0.99, full_matrices=False):
  """
  Peforms SVD and truncates the result to first r singular values
  using threshold provided.
  """
  U, S, Vh = np.linalg.svd(mat, full_matrices=full_matrices)
  r = len(S)
  s_cummulative = 0
  s_threshold = np.sum(S)*r_threshold
  for i, s in enumerate(S):
      s_cummulative+=s
      if s_cummulative > s_threshold:
          r = i+1
          break
  return (U[:,:r], S[:r,], Vh[:r,:], r)

