"""
Collection of classes to simulate time series of various systems
"""

# This file is to assist editor in autocompletion of the C++ module when used from python side.
from __future__ import annotations
from typing import Tuple
import numpy as np
__all__ = ['SSH_1D_satgain']
class SSH_1D_satgain:
  def __init__(self, N: int, psi0: float, satGainA: float, satGainB: float, gammaA: float, gammaB: float, time_end: float, time_delta: float, t1: float, t2: float) -> None:
    """
    1D SSH system with domain wall and saturated gain
    """
  def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the system using RK4 and returns the time series.
    This function releases GIL.
    Returns:
      Tuple[np.ndarray, np.ndarray]: tuple of results
      - matrix of amplitudes of system through time (N_site X N_time)
      - time values (N_time)
    """
