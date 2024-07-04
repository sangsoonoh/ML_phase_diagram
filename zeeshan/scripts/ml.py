# in this file I will code phase diagrams generated using dmd, admd, etc
import numpy as np
import h5py
import typer
from typing_extensions import Annotated
from typing import Optional, List
from pathlib import Path
import rich
import timeseries as ts
import src.basis as basis
import src.functions as f
import matplotlib.pyplot as plt
from rich.progress import Progress, TimeElapsedColumn, TextColumn, BarColumn, TimeRemainingColumn, ProgressColumn
import concurrent.futures

app = typer.Typer(help="To get started use generate command to make data file",
                  invoke_without_command=False,
)

class State:
  def __init__(self,):
    self.datafilepath = None
state = State()

def generate_phase_diagram_points(num_points, seed=42):
  np.random.seed(seed) #for reproducibility
  x_ = np.random.uniform(0, 0.5, num_points)
  y_ = np.random.uniform(0, 0.5, num_points)
  return x_,y_

@app.command()
def generate(
  numpoints:Annotated[Optional[int], typer.Option(help="Number of points on phase diagram")] = 100,
  seed:Annotated[Optional[int], typer.Option(help="Seed to use for random points on phase diagram for reproducability")] = 42,
  tsampsize:Annotated[Optional[int], typer.Option(help="Number of time points to sample")]=1000,
  ):
  """
  Generate data for system in SW's paper on 1D SSH with saturated gain.
  """
  x_,y_ = generate_phase_diagram_points(numpoints, seed=seed)

  tsfilename = str(state.datafilepath)
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

def j_star(Phis:List[np.ndarray], amplitudes_sample: np.ndarray):
  """
  Eq 7 in paper
  """
  projectors = [Phi @ np.linalg.pinv(Phi) for Phi in Phis]
  norms = [np.linalg.norm(projector @ amplitudes_sample) for projector in projectors]
  return np.argmax(norms)

@app.command('classify', help="Classify using fixed library")
def classify(
  indices:Annotated[List[int], typer.Argument(help="Indices in data file to use for fixed library. Comma-separated, no spaces e.g. 40,76")],
  basisMethod: Annotated[Optional[basis.BasisMethod], typer.Option("--basis", help="The method to use when producing basis for classification."),] = basis.BasisMethod.admd,
  N_aug: Annotated[Optional[int], typer.Option("--Naug", help="(When using aDMD) Augmentation factor; Note the augmented basis will be Naug+1 times taller")] = 5,
):
  with h5py.File(state.datafilepath, 'r+') as file:
    num_points = file.attrs['numpoints']
    library_datas = [np.asarray(file.get(f'{i}/amplitudes')) for i in indices]
    library_Phis = [basis.apply_basis_method(amplitudes=data, method=basisMethod, N_aug=N_aug) for data in library_datas]
    data_indices = np.arange(num_points)
    classifications = np.empty_like(data_indices, dtype=int)
    for i,_ in zip(data_indices, classifications):
      ith_data = file[f'{i}/amplitudes'][:,-N_aug-1:]
      if basisMethod == basis.BasisMethod.admd:
        ith_data = ith_data.flatten('F')
      classifications[i] = j_star(library_Phis, ith_data)
    f.update_hdf5(file, "classification/fixed-library", classifications, attrs={
      'basis': basisMethod.value
    })
    print(f"Classification performed: {classifications}, {classifications.shape}")

@app.command('plot-phases', help="Plot a phase diagram based on classification")
def plot(
):
  with h5py.File(state.datafilepath, 'r') as file:
    classifications = file.get("classification/fixed-library")
    num_points = file.attrs['numpoints']
    coords = np.array([(file.get(f'{i}').attrs['x'], file.get(f'{i}').attrs['y']) for i in np.arange(num_points)])
    fig, ax = plt.subplots()
    ax.scatter(coords[:,0], coords[:,1], marker='o', c=classifications)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_title('Phase diagram')
  return fig

@app.callback()
def main(datafilepath:Annotated[Path, typer.Argument(help="Path to HDF5 data file")]):
  state.datafilepath = str(datafilepath)

if __name__ == "__main__":
  app()