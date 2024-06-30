import timeseries as ts
import numpy as np
import concurrent.futures
import h5py
import typer
from rich.progress import Progress, TimeElapsedColumn, TextColumn, BarColumn, TimeRemainingColumn, ProgressColumn
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional

app = typer.Typer(no_args_is_help=True)

def generate_phase_diagram_points(num_points, seed=42):
  np.random.seed(seed) #for reproducibility
  x_ = np.random.uniform(0, 0.5, num_points)
  y_ = np.random.uniform(0, 0.5, num_points)
  return x_,y_

@app.command()
def generate_time_series_sw(
  out:Annotated[Path, typer.Argument(help="Path to *.hdf5 file to generate using this command")],
  numpoints:Annotated[Optional[int], typer.Option(help="Number of points on phase diagram")] = 100,
  seed:Annotated[Optional[int], typer.Option(help="Seed to use for random points on phase diagram for reproducability")] = 42,
  tsampsize:Annotated[Optional[int], typer.Option(help="Number of time points to sample")]=1000,
  ):
  """
  Generate data for system in SW's paper on 1D SSH with saturated gain.
  """
  x_,y_ = generate_phase_diagram_points(numpoints, seed=seed)

  tsfilename = str(out)
  h5py.File(tsfilename, 'w').close() #must create empty existing file to avoid overloading python memory
  time_sample_size = tsampsize
  with h5py.File(tsfilename, 'a') as tsfile:
    tsfile.attrs['seed'] = seed
    tsfile.attrs['tsampsize'] = tsampsize
    tsfile.attrs['numpoints'] = numpoints
    # todo: add more metadeta from the arguments to cpp below
    tsfile.flush()

  progress = Progress(
    TextColumn("{task.description} [progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    expand=False,
  )

  def time_series_worker(i, x, y, task):
    gamA = x
    system = ts.SSH_1D_satgain(N=10, psi0=0.01, satGainA=y+gamA, satGainB=0.,
                                    gammaA=gamA, gammaB=gamA, time_end=1200, time_delta=0.01,
                                    t1=1., t2=0.7)
    amplitudes, times = system.simulate()
    amplitudes = amplitudes[:,1799:] # start from 1800-th time step
    times = times[1799:]
    time_sample_filter = np.round(np.linspace(0,amplitudes.shape[0]-1,time_sample_size)).astype(int)
    with h5py.File(tsfilename, 'a') as file:
      grp = file.create_group(f"{i}")
      grp.create_dataset(name=f'amplitudes', data = amplitudes[:,time_sample_filter] )
      grp.create_dataset(name=f'times', data = times[time_sample_filter] )
      grp.attrs['x'] = x
      grp.attrs['y'] = y
      grp.attrs['N'] = 10
      file.flush()
    progress.update(task, advance=1, )
    return 'success'


  with progress:
    task = progress.add_task("Generating..", total=numpoints,  )
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(time_series_worker, i, x, y, task ) for i, (x,y) in enumerate(zip(x_,y_))]
      [future.result() for future in futures] 

if __name__ == "__main__":
  app()