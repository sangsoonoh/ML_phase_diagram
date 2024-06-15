from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path
from git import Repo

def download_eigen():
  eigen_dir = Path(__file__).resolve().parent / "deps" / "eigen"
  if not eigen_dir.is_dir():
    Repo.clone_from(R"https://gitlab.com/libeigen/eigen.git", eigen_dir)
  return str(eigen_dir)

eigen_include_dir = download_eigen()

# Define the extension module
ext_modules = [
  Pybind11Extension(
    "timeseries", 
    ["src/main.cpp"], 
    include_dirs=[eigen_include_dir],
    extra_compile_args=["-std=c++17", "-O2"],
  ),
]

setup(
  name="timeseries",
  version="0.1.0",
  author="Zeeshan Ahmad",
  description="Module for generating timeseries of various systems",
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
)