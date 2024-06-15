"""
Collection of classes to simulate time series of various systems
"""
from __future__ import annotations
import numpy
__all__ = ['SSH_1D_satgain']
class SSH_1D_satgain:
  def __init__(self, N: int, psi0: float, satGainA: float, satGainB: float, gammaA: float, gammaB: float, time_end: float, time_delta: float, t1: float, t2: float) -> None:
    ...
  def simulate(self) -> numpy.ndarray:
    """
    Time series of 1D SSH system with domain wall and saturated gain
    """
  def testfunc(self) -> str:
    ...
