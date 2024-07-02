import os
from pathlib import Path

package_name = "vqa_llm"

list_of_files = [
    ".github/workflows/ci.yaml",
    ".github/workflows/python_publish.yaml",
    ".github/workflows/update_hf.yaml",
    "src/__init__.py",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/inference/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/logger/__init__.py",
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/models/__init__.py",
    f"src/{package_name}/utils/__init__.py",
    "serving/flask_app/app.py",
    "serving/streamlit_app/app.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiments.ipynb",
    "Dockerfile",
    "docker-compose.yaml",

]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass # Create a empty file