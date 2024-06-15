import timeseries as ts
import numpy as np
import concurrent.futures
import h5py
import typer
app = typer.Typer(no_args_is_help=True)

def generate_phase_diagram_points(num_points):
  np.random.seed(42) #for reproducibility
  x_ = np.random.uniform(0, 0.5, num_points)
  y_ = np.random.uniform(0, 0.5, num_points)
  return x_,y_


def generate_time_series(export_file):
  x_,y_ = generate_phase_diagram_points(500)
  #generate data

  tsfilename = export_file
  h5py.File(tsfilename, 'w').close() #must create empty existing file to avoid overloading python memory
  time_sample_size = 1000

  def time_series_worker(i, x,y):
    gamA = x
    params = ts.SSH_1D_saturated_gain_Params(N=10, psi0=0.01, satGainA=y+gamA, satGainB=0.,
                                    gammaA=gamA, gammaB=gamA, time_end=1200, time_delta=0.01,
                                    t1=1., t2=0.7)
    time_series = ts.SSH_1D_saturated_gain(params)
    time_series = time_series[1799:,:] # start from 1800-th time step
    samp_indices = np.round(np.linspace(0,time_series.shape[0]-1,time_sample_size)).astype(int)
    with h5py.File(tsfilename, 'a') as tsfile:
      data = time_series[samp_indices,:]
      dataset = tsfile.create_dataset(name=f'series_{i}', data = data)
      dataset.attrs['x'] = x
      dataset.attrs['y'] = y
      dataset.attrs['N'] = 10
      tsfile.flush()
    return 'success'

  xy_pairs = np.array([(x,y) for x,y in zip(x_,y_)]) # this line maybe redundant against jsut zip(x,y)


  with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(time_series_worker, i, *(xy_pair)) for i, xy_pair in enumerate(xy_pairs)]

if __name__ == "__main__":
  app()