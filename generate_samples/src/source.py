import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import clear_output

#########################################
###    Define sources               ###
#########################################
def cw_source(A, omega, t):
    y = np.array([A * np.exp(1j*omega*t),    # A:  amplitude
                  A * np.exp(1j*omega*t)])   
    #y = A * np.exp(-1j*omega*t)    # A: amplitude
                   
    return y

def cw_probe_sources(A, A_probe, omega, t, omega2, tc, tau, N, xs_list ):
    source = np.zeros((2*N)).astype(complex)
    for i in xs_list:
        source[i] = np.sqrt(A) * np.exp(1j*omega*t) + np.sqrt(A_probe)*np.exp( -1* np.square((t-tc))/tau**2 )* np.exp(1j*omega2*t)
    
    return source 

def cw_sources(A, omega, t, N, xs_list):
    source = np.zeros((2*N)).astype(complex)
    for i in xs_list:
        source[i] = A 
       # source[i] = A * np.exp(1j*omega*t)
    
    return source 

def gaussian_sources(A, omega, tc, tau, t, N, xs_list):
    source = np.zeros((2*N)).astype(complex)
    for i in xs_list:
        source[i] = np.sqrt(A) * np.exp( -1* np.square((t-tc))/tau**2 )* np.exp(1j*omega*t)
    
    return source 


