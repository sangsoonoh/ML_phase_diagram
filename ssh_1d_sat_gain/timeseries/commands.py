import subprocess

def configure_timeseries():
  ps_code = """
  & .venv/Scripts/Activate.ps1
  & ./deps/conanbuild.ps1
  cmake -G Ninja -B timeseries/build -S ./timeseries
  """
  cmds = ['pwsh', '-Command', ps_code]
  r = subprocess.run(cmds)

def build_timeseries():
  ps_code = """
  & .venv/Scripts/Activate.ps1
  & ./deps/conanbuild.ps1
  cmake --build timeseries/build 
  """
  cmds = ['pwsh', '-Command', ps_code]
  r = subprocess.run(cmds, check=True)
  print(r)
  return 0
