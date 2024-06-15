import numpy as np

def calc_pod_modes(time_series, ):
  t_values = time_series[:,0].real.T
  psi_values = time_series[:,1:].T

  Ns,Nt = psi_values.shape
  N0 = np.min([Nt,Ns])
  N = int((Ns-1)/2)

  U, S, Vh = np.linalg.svd(psi_values, full_matrices=False)
  Phi = U
  return Phi, S, Vh