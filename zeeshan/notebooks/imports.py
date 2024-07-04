
from pathlib import Path
import sys
def find_scripts():
  print("22")
  current_file_path = Path(__file__).resolve()
  __folder = current_file_path.parent
  workspace = __folder.parent.parent
  scripts = workspace / "scripts"
  sys.path.append(str(scripts))
  print(sys.path)