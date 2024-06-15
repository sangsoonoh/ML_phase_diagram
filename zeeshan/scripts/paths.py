from pathlib import Path



current_file_path = Path(__file__).resolve()
current_directory = current_file_path.parent
workspace_directory = current_directory.parent