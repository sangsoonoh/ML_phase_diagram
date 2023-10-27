import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import clear_output
#########################################
###    Define Hamiltonians    ###
#########################################

def H(N, g_A, g_B, gamma_A, gamma_B, t1, t2, x):
    """
    Define Hamiltonians  dx/dt = H x, H = H0 + H_c + H_g  
    """

    M = 2*N+1 
    
    H_0 = 0 * np.identity(M)  

    H_c = np.zeros((M, M)).astype(complex) 

    for ii in range(M-1):
        if ii  < N:
            t_intra = t1
            t_inter = t2
        else:
            t_intra = t2
            t_inter = t1
  
        if(ii % 2 == 0): 
            H_c[ii,ii+1] = -1j*t_intra   
            H_c[ii+1,ii] = -1j*t_intra
            #H_c[N+ii,N+ii+1] = -1j*t_intra
            #H_c[N+ii+1,N+ii] = -1j*t_intra
        else:    
            H_c[ii,ii+1] = -1j*t_inter
            H_c[ii+1,ii] = -1j*t_inter
            #H_c[N+ii,N+ii+1] = -1j*t_inter
            #H_c[N+ii+1,N+ii] = -1j*t_inter

    H_g = np.zeros((M,M))
    for ii in range(M):
        if(ii % 2 == 0):
            H_g[ii,ii] = g_A / (1+ np.abs(x[ii])**2) - gamma_A
        else:
            H_g[ii,ii] = g_B / (1+ np.abs(x[ii])**2) - gamma_B

    H_tot = H_0 + H_c + H_g     
    return H_tot


#########################################
###    time integrate     ###
#########################################
def integrate_RK4(N, g_A, g_B, gamma_A, gamma_B, t1, t2, Nt, tab_t, dt):#, A_probe omega2, tc, tau,

    M = 2*N+1
    #S_save = np.zeros((2, Nt)).astype(complex)  
    x_save = np.zeros((M, Nt)).astype(complex)
    x = np.transpose(0.01*np.ones(M).astype(complex))

    for ii in range(Nt):              # RK4 method
        t = tab_t[ii]
   
        #S = cw_probe_sources(A, A_probe, omega, t, omega2, N, [0,N])
       
        #S = gaussian_sources(A, omega, tc,tau, t, N, [0,N])
        #S = cw_probe_sources(A, omega, t, N, [0,N] )#, A_probe, omega2, tc, tau
        #S = cw_sources(A, omega0, t, N, [0,N] )#, A_probe, omega2, tc, tau
        #S_save[0,ii]= S[0]
        #S_save[1,ii]= S[N]  

        k1 = np.dot(H(N, g_A, g_B, gamma_A, gamma_B, t1, t2, x), x) 
        #    + cw_sources(A, omega0, t, N, [0,N] )#, A_probe, omega2, tc, tau
        k2 = np.dot(H(N, g_A, g_B, gamma_A, gamma_B, t1, t2, x + 0.5*k1*dt), x + 0.5*k1*dt) 
        #    + cw_sources(A, omega0, t, N, [0,N] )#, A_probe, omega2, tc, tau
        k3 = np.dot(H(N, g_A, g_B, gamma_A, gamma_B, t1, t2, x + 0.5*k2*dt), x + 0.5*k2*dt) 
        #    + cw_sources(A, omega0, t, N, [0,N] )#, A_probe, omega2, tc, tau
        k4 = np.dot(H(N, g_A, g_B, gamma_A, gamma_B, t1, t2, x + k3*dt), x + k3*dt) 
        #    + cw_sources(A, omega0, t, N, [0,N] )#, A_probe, omega2, tc, tau
   
        x = x + (1./6) * (k1 + 2*(k2 + k3) + k4) * dt
   
        x_save[:,ii] = x

    return x_save

#########################################
###    time integrate    cw source    ###
#########################################

def g_scan(tab_g, N, g_A, g_B, gamma_A, gamma_B, t_intra, t_inter, Nt, tab_t,dt, x1, x2):
    #S_save = np.zeros((2, Nt)).astype(complex)  
    x_save1 = np.zeros((Nsweep, 2*N, Nt )).astype(complex)
    x = np.transpose(np.zeros((2*N)).astype(complex))
   
    x[0] = x1
    x[N] = x2
    for iA in range(Nsweep):
        clear_output(wait=True)
        print(f"Iter: {iA+1}/{Nsweep}")
        x_save = integrate_RK4(N, g_A, g_B, gamma_A, gamma_B, t_intra, t_inter, N, Nt, tab_t,dt, x1, x2)
        x_save1[iA,:,:] = x_save

    return x_save1



def B_scan(A, omega_0, eta, theta, A_, tab_B_, N, Nsweep, Nt, tab_t,dt, x1, x2):
    S_save = np.zeros((2, Nt)).astype(complex)
    x_save1 = np.zeros((Nsweep, 2*N, Nt )).astype(complex)
    x = np.transpose(np.zeros((2*N)).astype(complex))

    x[0] = 0.11
    x[N] = 0.04
    for iB in range(Nsweep):
        clear_output(wait=True)
        print(f"Iter: {iB+1}/{Nsweep}")
        S_save,x_save = integrate_RK4(A, omega_0,eta,theta, A_,tab_B_[iB], N, Nt, tab_t,dt, x1, x2)
        x_save1[iB,:,:] = x_save

    return x_save1
#########################################
###    Plot functions             ###
#########################################

def plot_source(tab_t,S_save,t_max):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5))
    ax1.plot(tab_t,S_save[0,:])
    ax1.plot(tab_t,S_save[1,:])
    # plt.plot(tab_t,S_save[0,:])

    ax1.set_xlim([0,t_max])
    ax1.set_xlabel("time")
    ax1.set_ylabel("Source field amplitude")

    ax2.plot(tab_t,S_save[0,:])
    ax2.plot(tab_t,S_save[1,:])
    # plt.plot(tab_t,S_save[0,:])

    ax2.set_xlim([t_max-30,t_max])
    ax2.set_xlabel("time")
    ax2.set_ylabel("Source field amplitude")   
    plt.show()

def plot_amplitude(tab_t,x_save,t_max, y_max, N):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 5))
    
    for ii in range(2*N): 
        ax1.plot(tab_t,x_save[ii,:])
        ax1.set_xlim([0,t_max])
        # plt.xlim([t_max-30,t_max])
        ax1.set_xlabel("time")
        ax1.set_ylabel("Field amplitude")
 

    for ii in range(2*N): 
        ax2.plot(tab_t,x_save[ii,:])
        ax2.set_xlim([t_max-100,t_max])
        ax2.set_ylim([-y_max,y_max])
        ax2.set_xlabel("time")
        ax2.set_ylabel("Field amplitude")
    plt.show()
