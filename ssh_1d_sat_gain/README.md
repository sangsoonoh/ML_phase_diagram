## [`ssh_1d_sat_gain`](https://github.com/sangsoonoh/ML_phase_diagram/tree/main/ssh_1d_sat_gain)
  - Recreate some results from [Wong, S., Olthaus, J., Bracht, T.K. et al., 10.1038/s42005-023-01230-z](https://www.nature.com/articles/s42005-023-01230-z) for practice.

# Development
## Project organisation
### Tools:
- [Python 3.12](https://www.python.org/): using `venv` is recommended;  `requirements.txt`
- [Conan 2](https://conan.io/): `conanfile.txt`
- C++ compiler tools, `cmake`, `ninja`
- C++ libraries: pybind11, 
- `Powershell`

### Files
- Jupyter notebook files are the ultimate client code, presenting results of calculations.
- `ssh_1d/functions.py` module contains refactored python functions relevant to 1D SSH model.
- `ssh_1d/functions.cpp` contains additional C++ functions available to `functions.py` and the notebooks when compiled using the steps below.

### Using project manually without scripts
Clone this repo, install powershell, cd into repo, and then in powershell:

1. Create a python venv `python -m venv .venv` : The dir `.venv` is git ignored.
1. (A) Activate the python venv `./venv/Scripts/activate` : May need to set execution policy to allow script execution, e.g. `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`.
1. Install python packages in the venv `pip install -r requirements.txt`.
1. Install conan packages in `deps` dir `conan install . of deps --conf=tools.env.virtualenv:powershell=True` : It may ask you to run `conan profile detect` first. 
1. (A) Update path environment variable by running `deps/conanbuild.ps1`.
1. (B) Generate project for C++ module `cmake --preset=functions -B ssh_1d/build -S ssh_1d`.
1. (C) Build generated project `cmake -S ssh_1d -B ssh_1d/build --preset=functions`. 

Steps above generally do not need repeating, except the ones marked:
- (A) repeated every time powershell restarts.
- (B) repeated every time build directory removed or clean.
- (C) repeated every time C++ code changes or build directory removed.

### Helper commands to repeat steps (B) and (C)
First run `pip install -e .\commands\`. Then we have the following convenience commands:
- (B) `configure-cpp-functions`
- (C) `build-cpp-functions`.

To clean C++ project, delete `ssh_1d/build` folder.