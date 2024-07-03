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

def print_hdf5(path:str):
  """
  Lists the items in hdf5 file with their attrs
  """
  def custom_print(name, obj):
    attrs = {key:value for key,value in obj.attrs.items()}
    rich.print(f"{name}, {obj}, {attrs}")

  with h5py.File(path) as file:
    custom_print("/", file)
    file.visititems(custom_print)

def update_hdf5(file:h5py.File, link:str, data:np.ndarray, attrs:dict = None):
  """
  Updates dataset at given link path. Creates data set if it does not exist.
  Assumes that data is the same size as existing data
  """
  if not (file.get(link)):
    file.create_dataset(link, shape=data.shape)
  file[link][:] = data
  if attrs:
    for key,value in attrs.items():
      file[link].attrs[key] = value
