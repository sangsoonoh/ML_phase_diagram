import timeseries
import rich

print(timeseries)

system = timeseries.SSH_1D_satgain(N=10, psi0=0.01, satGainA=0.1, satGainB=0.,
                                gammaA=.1, gammaB=.1, time_end=1200, time_delta=0.01, t1=1., t2=0.7)
