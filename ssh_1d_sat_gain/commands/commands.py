import typer
import subprocess
import sys
import os

def main():
  print("hello worlds")
def cli():
  print("print from run")
def initenv():
  #r = subprocess.Popen(f'pwsh.exe -NoProfile -ExecutionPolicy Bypass -file "./deps/conanbuild.ps1"', stdout=sys.stdout, shell=True)
  r = subprocess.run(R'deps\conanbuild.bat', shell=True)
  print(r.stdout)
  typer.echo("C++ tools now available")

def configure_cpp():
  ps_code = """
  & .venv/Scripts/Activate.ps1
  & ./deps/conanbuild.ps1
  cmake --preset=functions -B ssh_1d/build -S ssh_1d
  """
  cmds = ['pwsh', '-Command', ps_code]
  r = subprocess.run(cmds)

def build_cpp():
  ps_code = """
  & .venv/Scripts/Activate.ps1
  & ./deps/conanbuild.ps1
  cmake -S ssh_1d -B ssh_1d/build --preset=functions
  """
  cmds = ['pwsh', '-Command', ps_code]
  r = subprocess.run(cmds, check=True)
  print(r)
  return 0

if __name__ == "__main__":
  typer.run(main)