#convenience variables storing absolute paths to places in project

from pathlib import Path

current_file_path = Path(__file__).resolve()
__folder = current_file_path.parent
workspace = __folder.parent.parent