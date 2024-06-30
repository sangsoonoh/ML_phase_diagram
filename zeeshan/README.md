  ## References
  - [Wong, S., Olthaus, J., Bracht, T.K. et al., 10.1038/s42005-023-01230-z](https://www.nature.com/articles/s42005-023-01230-z) 

# Using this workspace
### Tools:
- [Python 3.12](https://www.python.org/): using `venv` is recommended;  see `requirements.txt` for python packages used
- C++ compiler installed and available in PATH

Clone this repo, cd into repo, and then the following commands:

1. Create a python venv `python -m venv .venv` : The dir `.venv` is git ignored.
1. Activate the python venv `./venv/Scripts/activate`
1. Install python packages in the venv `pip install -r requirements.txt`.
1. `pip install wheel setuptools pip --upgrade` (if next step says it can't find pybind11)
1. Install `timeseries` python package `pip install .\timeseries\` (every time C++ needs recompiling)

### Folders:
- `scripts\`: Python scripts meant to be ran directly
- `timeseries\`: C++ pybind module for producing timeseries for different systems
- `data\`: For storing generated timeseries datas (hdf5 files are gitignored)
- `output\`: For storing results like plots, etc
- `notebooks\`: Jupyter notebooks showing examples, etc

## Scripts
The following scripts are `typer` apps, so run them with `--help` flag for info on arguments, options etc.

### `generate.py`
Command for simulating time series and generating data files.
E.g. `python ./scripts/generate.py --seed 42 ./data/example.hdf5` to generate a data file.
Run `python ./scripts/generate.py --help` for all command options.

### `classify.py`
Makes data for a phase diagram based on chosen basis method (e.g. `dmd`) and stores it into the same data file it read from.
