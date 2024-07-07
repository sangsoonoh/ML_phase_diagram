"""
Wrapper to standardise data file structure
"""
import h5py
import numpy as np
from typing import Tuple, List
from concurrent.futures import Future
import src.classifier as cl

Times = np.ndarray
Amplitudes = np.ndarray
Metadata = dict


class DataFile:

  def __init__(self, path:str,) -> None:
    self.path = path
    h5py.File(path, 'a').close()

  def __str__(self,)->str:
    """
    Lists the items in hdf5 file with their attrs
    """
    lines = []
    def onvisit(name, obj):
      attrs = self.__attrs_to_dict(obj)
      lines.append(f"{name}, {obj}, {attrs}")

    with h5py.File(self.path) as file:
      onvisit("/", file)
      file.visititems(onvisit)
    return "\n".join(lines)
  
  def clear(self, ):
    """
    Clears the data file
    """
    with h5py.File(self.path, 'w') as file:
      file.clear()
  
  def writemany_timeseries(self, timeseries_list: List[Tuple[Amplitudes, Times, Metadata]] | List[Future[Tuple[int, Amplitudes, Times, Metadata]]]):
    """
    This function can be used with concurrent's ThreadPoolExecutor
    Arguments:
      - series_at: List of (index,amplitudes, times, attrs). Can also be a list of concurrent's
        Futures that return a tuple of (index,amplitudes, times, attrs). 
        Note: the Futures will be awaited in the list order from the beginning.
    """
    for timeseries in timeseries_list:
      if isinstance(timeseries, Future):
        index, amplitudes, times, attrs = timeseries.result()
        self.write_timeseries(index, amplitudes, times, attrs)
      elif isinstance(timeseries, Tuple):
        self.write_timeseries(*timeseries)
  
  def readmany_timeseries(self, indices:List[int] = []):
    """
    Generator function to loop over timeseries e.g. for index,amplitudes,times,attrs in df.getmany_timeseries():
    Arguments:
      - indices: Optional list of indices to yield. Yields all if None
    """
    if len(indices) == 0:
      indices = self.__all_timeseries_indices()
    for index in indices:
      yield index, *self.read_timeseries(index) 


  def write_timeseries(self, index:int, amplitudes:np.ndarray, times:np.ndarray, attrs:dict = None):
    with h5py.File(self.path, 'a') as file:
      if not isinstance(file.get(f"timeseries/{index}"),h5py.Group):
        grp = file.create_group(f"timeseries/{index}")
      if attrs:
        self.__dict_to_attrs(grp, attrs)
      grp.create_dataset("amplitudes", data=amplitudes)
      grp.create_dataset("times", data=times)

  def read_timeseries(self, index:int)->Tuple[Amplitudes,Times,Metadata]:
    with h5py.File(self.path, 'r') as file:
      times = np.asarray(file.get(f"timeseries/{index}/times"))
      amplitudes = np.asarray(file.get(f"timeseries/{index}/amplitudes"))
      attrs = self.__attrs_to_dict(file.get(f"timeseries/{index}"))
    return amplitudes, times,attrs

  def write_classification(self, classifyMethod:cl.ClassifyMethod, basisMethod:cl.BasisMethod, data:np.ndarray, attrs:Metadata):
    with h5py.File(self.path, 'a') as file:
      dsname = f"classifications/{classifyMethod}/{basisMethod}"
      if dsname in file:
        file[dsname][:] = data
      else:
        file.create_dataset(dsname, data=data)
      self.__dict_to_attrs(file[dsname], attrs)
        
  def read_classification(self, classifyMethod:cl.ClassifyMethod, basisMethod:cl.BasisMethod, ):
    with h5py.File(self.path, 'r') as file:
      dsname = f"classifications/{classifyMethod}/{basisMethod}"
      data = np.asarray(file[dsname])
      attrs = self.__attrs_to_dict(file[dsname])
    return data, attrs

  @property
  def metadata(self):
    attrs = None
    with h5py.File(self.path, 'r') as file:
      attrs = self.__attrs_to_dict(file)
    return attrs
  
  @metadata.setter
  def metadata(self, newattrs):
    with h5py.File(self.path, 'a') as file:
      for key,_ in file.attrs.items():
        del file.attrs[key]
    self.update_metadata(newattrs)
  def update_metadata(self, newattrs):
    with h5py.File(self.path, 'a') as file:
      for key, value in newattrs.items():
        file.attrs[key] = value

  def update(self, link:str, data:np.ndarray, attrs:dict = None):
    """
    Updates dataset at given link path. Creates data set if it does not exist.
    Assumes that data is the same size as existing data
    """
    with h5py.File(self.path, 'a') as file:
      if not (file.get(link)):
        file.create_dataset(link, shape=data.shape)
      file[link][:] = data
      if attrs:
        for key,value in attrs.items():
          file[link].attrs[key] = value

  def __attrs_to_dict(self, obj):
    return {key:value for key,value in obj.attrs.items()}
  def __dict_to_attrs(self, obj, attrs:dict)->None:
    for key,value in attrs.items():
      obj.attrs[key] = value
  def __all_timeseries_indices(self,)->List[str]:
    with h5py.File(self.path, 'r') as file:
      listofindices = [ int(group) for group in file.get("timeseries")]
    return listofindices