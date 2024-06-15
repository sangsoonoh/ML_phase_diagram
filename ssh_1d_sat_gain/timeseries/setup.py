from pathlib import Path
from setuptools import setup
import glob

subdirectory = 'build'  # Change this to your actual subdirectory name
current_directory = Path(__file__).parent

search_pattern = current_directory / subdirectory / '*.pyd'
pyd_files = list(glob.glob(str(search_pattern)))

if not pyd_files:
  raise FileNotFoundError("No .pyd file found in the specified subdirectory.")

pyd_file_path = pyd_files[0]
print(pyd_file_path)
relative_path = Path(pyd_file_path).relative_to(current_directory)
print("before setup")
setup(
    name='timeseries',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'configure-timeseries=commands:configure_timeseries',
            'build-timeseries=commands:build_timeseries',
        ]
    },
    ext_package= str(relative_path) 
)