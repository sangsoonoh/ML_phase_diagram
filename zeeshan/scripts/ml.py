# in this file I will code phase diagrams generated using dmd, admd, etc
import numpy as np
import h5py
import typer
from typing_extensions import Annotated
from typing import Optional, List
from pathlib import Path
import rich
import timeseries as ts
import src.classifier as classifier
import src.functions as f
import src.datafile as datafile
import matplotlib.pyplot as plt
from rich.progress import Progress, TimeElapsedColumn, TextColumn, BarColumn, TimeRemainingColumn, ProgressColumn
import concurrent.futures

app = typer.Typer(help="To get started use generate command to make data file",
                  invoke_without_command=False,
)

class State:
  def __init__(self,):
    self.datafilepath = None
    self.plotIndices = None
state = State()

def generate_phase_diagram_points(num_points, seed=42):
  np.random.seed(seed) #for reproducibility
  x_ = np.random.uniform(0, 0.5, num_points)
  y_ = np.random.uniform(0, 0.5, num_points)
  return x_,y_

@app.command()
def generate(
  numpoints:Annotated[Optional[int], typer.Option("--points",help="Number of points on phase diagram")] = 100,
  seed:Annotated[Optional[int], typer.Option(help="Seed to use for random points on phase diagram for reproducability")] = 42,
  tsampsize:Annotated[Optional[int], typer.Option(help="Number of time points to sample")]=2000,
  ):
  """
  Generate data for system in SW's paper on 1D SSH with saturated gain.
  """
  x_,y_ = generate_phase_diagram_points(numpoints, seed=seed)

  df = datafile.DataFile(str(state.datafilepath),)
  df.clear()
  df.metadata = {
    'seed' : seed,
    'tsampsize': tsampsize,
    'numpoints': numpoints,
  }

  time_sample_size = tsampsize

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
                                    gammaA=gamA, gammaB=gamA, time_end=1400, time_delta=0.01,
                                    t1=1., t2=0.7)
    amplitudes, times = system.simulate()

    time_sample_filter = np.round(np.linspace(0,times.shape[0]-1,time_sample_size)).astype(int)

    amplitudes = amplitudes[:,time_sample_filter]
    times = times[time_sample_filter]

    amplitudes = amplitudes[:,1800:]
    times = times[1800:]

    progress.update(task, advance=1, )
    attrs = { 'x':x, 'y':y, }
    #return i,  amplitudes, times, attrs
    return i,  amplitudes, times, attrs

  with progress:
    task = progress.add_task("Generating..", total=numpoints,  )
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(time_series_worker, i, x, y, task ) for i, (x,y) in enumerate(zip(x_,y_))]
      df.writemany_timeseries(futures)

def j_star(Phis:List[np.ndarray], amplitudes_sample: np.ndarray):
  """
  Eq 7 in paper
  """
  projectors = [Phi @ np.linalg.pinv(Phi) for Phi in Phis]
  projecteds = [projector @ amplitudes_sample for projector in projectors]
  norms = [np.linalg.norm(projected,2) for projected in projecteds]
  return np.argmax(norms)

@app.command('classify', help="Classify using fixed library")
def classify(
  indices:Annotated[List[int], typer.Argument(help="Indices in data file to use for fixed library. Comma-separated, no spaces e.g. 40,76")],
  classifyMethod: Annotated[Optional[classifier.ClassifyMethod], typer.Option("--method", help="Classification method")],
  basisMethod: Annotated[Optional[classifier.BasisMethod], typer.Option("--basis", help="The method to use when producing basis for classification."),] = classifier.BasisMethod.admd,
  N_aug: Annotated[Optional[int], typer.Option("--Naug", help="(When using aDMD) Augmentation factor; Note the augmented basis will be Naug+1 times taller")] = 5,
):
  df = datafile.DataFile(state.datafilepath)
  num_points = df.metadata['numpoints']
  library_datas = [df.read_timeseries(i)[0] for i in indices]
  library_Phis = [classifier.apply_basis_method(amplitudes=data, method=basisMethod, N_aug=N_aug) for data in library_datas]
  classifications = np.arange(num_points)
  for index,amplitudes,times,attrs in df.readmany_timeseries():
    ith_data = amplitudes[:,-N_aug-1:]
    if basisMethod == classifier.BasisMethod.admd:
      ith_data = ith_data.flatten('F')
    classifications[index] = j_star(library_Phis, ith_data)
  df.write_classification(classifyMethod, basisMethod, classifications, { 'indices': indices })
  print(f"Classification performed: {classifications}, {classifications.shape}")

@app.command('plot-phases', help="Plot a phase diagram based on classification")
def plot(
  classifyMethod: Annotated[Optional[classifier.ClassifyMethod], typer.Option("--method", help="Classification method")],
  basisMethod: Annotated[Optional[classifier.BasisMethod], typer.Option("--basis", help="The method to use when producing basis for classification."),] = classifier.BasisMethod.admd,
  show: Annotated[Optional[bool], typer.Option(help="Show the matplotlib plot before exiting")] = False,
  save: Annotated[Optional[Path], typer.Option(help="Path to save figure to")] = None,
):
  df = datafile.DataFile(state.datafilepath)
  classifications,clattrs = df.read_classification(classifyMethod, basisMethod)
  indices,x,y = zip(*[ (index, attrs['x'], attrs['y']) for index,*_, attrs in df.readmany_timeseries()])
  state.plotIndices = indices
  fig, ax = plt.subplots()
  ordered_classifications = [classifications[index] for index in indices]
  ax.scatter(x, y, marker='o', c=ordered_classifications, picker=True)
  ax.set_xlabel('x')
  ax.set_xlabel('y')
  ax.set_title('Phase diagram')
  x,y = zip(*[ (attrs['x'], attrs['y']) for *_, attrs in df.readmany_timeseries(clattrs['indices'])])
  ax.scatter(x, y, marker='x', c='red', s = 20, label=f'library')
  ax.legend()
  if show:
    def on_pick(event):
      print(state.plotIndices[event.ind[0]])
    plt.gcf().canvas.mpl_connect('pick_event', on_pick)
    plt.show()
  if save:
    fig.savefig(str(save),)
  return fig


@app.callback()
def main(datafilepath:Annotated[Path, typer.Argument(help="Path to HDF5 data file")]):
  state.datafilepath = str(datafilepath)

if __name__ == "__main__":
  app()