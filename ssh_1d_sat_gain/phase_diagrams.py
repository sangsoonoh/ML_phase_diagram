# in this file I will code phase diagrams generated using dmd, admd, etc
import numpy as np
import matplotlib.pyplot as plt
import ssh_1d.functions as functions
import h5py

def classify_eq7(Phis, data):
  '''
  Clasifies a given time series data using the set of bases {Phi_j}
  Phi_j is a Nw x Nw matrix (check this)
  data is Nw x Nw matrix (check this)
  '''
  pass 


tsfilename = 'data/time_series.hdf5'

with h5py.File(tsfilename, 'r') as file:
  num_points = 500 # number of points on phase diagram
  library_indices = [4,11]
  library_datas = [file[f'series_{i}'][:] for i in library_indices]
  library_Phis = [functions.calc_pod_modes(time_series=data)[0] for data in library_datas]
  library_Ps = [Phi @ Phi.H for Phi in library_Phis]
  data_indices = np.arange(num_points)
  classifications = np.empty_like(data_indices, dtype=int)
  for i,_ in zip(data_indices, classifications):
    ith_data = file[f'series_{i}'][:]
    norms = [np.linalg.norm(Pj @ ith_data) for Pj in library_Ps]
    j_star = np.argmax(norms)
    classifications[i] = j_star


