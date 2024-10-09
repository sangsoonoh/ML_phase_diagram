# in this file I will code phase diagrams generated using dmd, admd, etc
from enum import Enum
from click import Context
import numpy as np
import rich.progress
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
from typer.core import TyperGroup
import networkx as nx
import random

class OrderCommands(TyperGroup):
  def list_commands(self, ctx: Context) -> List[str]:
    return list(self.commands)

app = typer.Typer(help="To get started try commands below with --help option",
                  invoke_without_command=False,
                  cls=OrderCommands,
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

@app.command('classify', help="Classify time series from a library")
def classify(
  libraryName: Annotated[str, typer.Option("--library", help="Name of library to use for classification. See library command")],
):
  df = datafile.DataFile(state.datafilepath)
  library_indices, library_attrs = df.read_library(libraryName)
  basisMethod = classifier.BasisMethod(library_attrs['basis'])
  N_aug = library_attrs['Naug']
  library = [
    classifier.apply_basis_method(amplitudes=amplitudes, method=basisMethod, N_aug=N_aug)
    for _, amplitudes, *_ in df.readmany_timeseries(library_indices)
  ]

  num_points = df.metadata['numpoints']
  classifications = np.arange(num_points)
  for index,amplitudes,*_ in df.readmany_timeseries():
    ith_data = amplitudes[:,-N_aug-1:]
    if basisMethod == classifier.BasisMethod.admd:
      ith_data = ith_data.flatten('F')
    classifications[index] = j_star(library, ith_data)
  df.write_classification(libraryName, classifications, { 'indices': library_indices, 'library': libraryName })
  print(f"Classification performed: {classifications}, {classifications.shape}")


class LibraryMethod(str, Enum):
  fixed="fixed"
  topdown="topdown"
  bottomup="bottomup"

@app.command('library', help="Make and add a library to the timeseries data")
def library(
  libraryName: Annotated[str, typer.Option("--name", help="Name of library to create")],
  libraryMethod: Annotated[LibraryMethod, typer.Option("--method", help="Library creation method")],
  fromIndexSet: Annotated[str, typer.Option("--from", help="List of comma-separated indices, name of another library or 'random,n'")],
  basisMethod: Annotated[Optional[classifier.BasisMethod], typer.Option("--basis", help="Basis method"),] = classifier.BasisMethod.admd,
  N_aug: Annotated[Optional[int], typer.Option("--Naug", help="(When using aDMD basis method) Augmentation factor - 1")] = 5,
  seed:Annotated[Optional[int], typer.Option(help="Random seed")] = 30,
  gamma_threshold:Annotated[Optional[float], typer.Option("--gamma", help="Gamma threshold; only used with top-down library method")] = 0.75,
):
  df = datafile.DataFile(state.datafilepath)
  attrs = { 
    'basis': str(basisMethod), 
    'Naug': int(N_aug), 
    'method': basisMethod.value, 'from': str(fromIndexSet) ,
    'gamma': float(gamma_threshold),
  }
  #parse --from:
  try:
    from_library_indices, attrs = df.read_library(fromIndexSet)
  except:
    if fromIndexSet.startswith("random"):
      n = int(fromIndexSet.split(",")[1])
      random.seed(seed)
      from_library_indices = random.sample(range(df.metadata['numpoints']), n)
    else:
      from_library_indices = [int(i) for i in fromIndexSet.split(",")]

  match libraryMethod:
    case LibraryMethod.fixed:
      df.write_library(libraryName, from_library_indices, attrs=attrs)
    case LibraryMethod.topdown:
      from_library = [
        classifier.apply_basis_method(amplitudes=amplitudes, method=basisMethod, N_aug=N_aug)
        for _, amplitudes, *_ in df.readmany_timeseries(from_library_indices)
      ]
      groups = top_down_group_library(from_library, gamma_threshold)
      groups_in_ts = [[from_library_indices[i] for i in group] for group in groups]
      representatives = [random.choice(choices) for choices in groups_in_ts]
      df.write_library(libraryName, representatives, attrs=attrs)
    case LibraryMethod.bottomup:
      pass
      

def gamma_ij(projector1:np.ndarray, projector2:np.ndarray)->np.float64:
  return np.linalg.norm(projector1 @ projector2, 'fro')**2 / (np.linalg.norm(projector1, 'fro')*np.linalg.norm(projector2,'fro'))

def projector(Phi:np.ndarray):
  return Phi @ np.linalg.pinv(Phi)


def top_down_group_library(library:List[np.ndarray], gamma_th:float) -> List[List[int]]:
  G = nx.Graph()
  n = len(library)
  G.add_nodes_from(nodes_for_adding=np.arange(n))

  with rich.progress.Progress(transient=True) as progress:
    projectors = [projector(Phi) for Phi in library]
    edges = []
    edges_task = progress.add_task("[purple]Building library reduction graph", total=int(n*(n-1)/2),)
    for i in np.arange(n):
      for j in np.arange(i+1,n):
        progress.update(edges_task, advance=1)
        g = gamma_ij(projectors[i], projectors[j])
        if g > gamma_th:
          edges.append((i,j))
    G.add_edges_from(edges)
  
  groups = [list(component) for component in nx.find_cliques(G,)]
  n_groups = len(groups)
  for i in np.arange(n_groups):
    for j in np.arange(i+1, n_groups):
      if len(groups[j]) > len(groups[i]):
        j,i = (i,j)
      groups[i] = [item for item in groups[i] if item not in groups[j]]
  return groups



# @app.command('plot-phases', help="Plot a phase diagram based on classification")
# def plot_phases(
#   classifyMethod: Annotated[Optional[classifier.ClassifyMethod], typer.Option("--method", help="Classification method")],
#   basisMethod: Annotated[Optional[classifier.BasisMethod], typer.Option("--basis", help="The method to use when producing basis for classification."),] = classifier.BasisMethod.admd,
#   show: Annotated[Optional[bool], typer.Option(help="Show the matplotlib plot before exiting")] = False,
#   save: Annotated[Optional[Path], typer.Option(help="Path to save figure to")] = None,
# ):
#   """
#   Plot a phase diagram based on classification
#   """
#   df = datafile.DataFile(state.datafilepath)
#   classifications,clattrs = df.read_classification(classifyMethod, basisMethod)
#   indices,x,y = zip(*[ (index, attrs['x'], attrs['y']) for index,*_, attrs in df.readmany_timeseries()])
#   state.plotIndices = indices
#   fig, ax = plt.subplots()
#   ordered_classifications = [classifications[index] for index in indices]
#   ax.scatter(x, y, marker='o', c=ordered_classifications, picker=True)
#   ax.set_xlabel('x')
#   ax.set_xlabel('y')
#   ax.set_title('Phase diagram')
#   x,y = zip(*[ (attrs['x'], attrs['y']) for *_, attrs in df.readmany_timeseries(clattrs['indices'])])
#   ax.scatter(x, y, marker='x', c='red', s = 20, label=f'library')
#   ax.legend()
#   if show:
#     def on_pick(event):
#       print(state.plotIndices[event.ind[0]])
#     plt.gcf().canvas.mpl_connect('pick_event', on_pick)
#     plt.show()
#   if save:
#     fig.savefig(str(save),)
#   return fig


@app.callback()
def main(hdf5_path:Annotated[Path, typer.Argument(help="Path to HDF5 data file", show_default=False)]):
  state.datafilepath = str(hdf5_path)

if __name__ == "__main__":
  app()