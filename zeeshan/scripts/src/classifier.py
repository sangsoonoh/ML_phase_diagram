"""
Functions to calculate basis from time series data.
"""
import numpy as np
import src.functions as f
from enum import Enum

class BasisMethod(str, Enum):
  pod = "pod"
  dmd = "dmd"
  admd = "admd"

def pod(amplitudes:np.ndarray,):
  """
  Calculates POD basis from time series
  Arguments:
    - amplitudes: 2D matrix (N_site X N_time) containing history of amplitudes in the system
  Returns:
    - Phi: basis matrix whose columns are DMD modes (N_site X N_r)
  """
  psi_values = amplitudes

  U, S, Vh, r = f.trunc_svd(psi_values,)
  Phi = U
  return Phi, S, Vh

def dmd(amplitudes:np.ndarray):
  """
  Calculates DMD modes from time series

  Arguments:
    - amplitudes: 2D matrix (N_site X N_time) containing history of amplitudes in the system

  Returns:
    - Phi: basis matrix whose columns are DMD modes (N_site X N_r)
    - lam: corresponding eigenvalues of DMD modes (N_r)
  """
  psi_values = amplitudes
  # step 1
  X1 = psi_values[:,:-1]
  X2 = psi_values[:,1:]

  #step 2
  U,S,Vh,r = f.trunc_svd(X1,)
  Smat = np.diag(S)
  #step 3
  invS = np.linalg.inv(Smat)
  A_r = U.conj().T @ X2 @ Vh.conj().T @ invS
  
  #step 4
  lam, W = np.linalg.eig(A_r)

  #step 5
  Phi = X2 @ Vh.conj().T @ invS @ W

  return (Phi, lam,)

def augment_dmd(Phi:np.ndarray, lam:np.ndarray, N_aug:int):
  """
  time-augments DMD modes using corresponding eigenvalues

  Arguments:
    - Phi: basis matrix whose columns are DMD modes (N_site X N_r)
    - lam: corresponding eigenvalues of DMD modes (N_r)

  Returns:
    - aPhi: augmented DMD modes ((N_aug+1)*N_site X N_r)
  """
  Ns,Nr = Phi.shape
  Lam = np.tile(lam, (Ns,1))
  aShape = (Ns*(N_aug+1), Nr)
  aPhi = np.zeros(aShape, dtype='complex')
  for n in np.arange(N_aug+1):
      aPhi[n*Ns:(n+1)*Ns, :] = Lam**n * Phi
  return aPhi

def apply_basis_method(amplitudes:np.ndarray, method:BasisMethod, N_aug:int=5):
  match method:
    case BasisMethod.pod:
      return pod(amplitudes)[0]
    case BasisMethod.dmd:
      return dmd(amplitudes)[0]
    case BasisMethod.admd:
      return augment_dmd(*dmd(amplitudes), N_aug=N_aug)

class ClassifyMethod(str, Enum):
  fixed_library = "fixed-library"