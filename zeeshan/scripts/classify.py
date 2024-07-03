# in this file I will code phase diagrams generated using dmd, admd, etc
import numpy as np
import h5py
import typer
from typing_extensions import Annotated
from typing import Optional, List
from pathlib import Path
import rich
import src.basis as basis
import src.functions as f

app = typer.Typer()
class State:
  def __init__(self,):
    self.datafilepath = None
state = State()


def j_star(Phis:List[np.ndarray], amplitudes_sample: np.ndarray):
  """
  Eq 7 in paper
  """
  projectors = [Phi @ np.linalg.pinv(Phi) for Phi in Phis]
  norms = [np.linalg.norm(projector @ amplitudes_sample) for projector in projectors]
  return np.argmax(norms)

@app.command('fixed-library', help="Classify using fixed library")
def fixed_library(
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


@app.callback()
def main(datafilepath:Annotated[Path, typer.Argument(help="path to hdf5 timeseries data file")]):
  state.datafilepath = str(datafilepath)

if __name__ == "__main__":
  app()