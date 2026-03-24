import os
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO , format = '[%(asctime)s]: %(message)s:')

project_name = "medChat"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/prompts.py",
    f"{project_name}/config.py",
    f"{project_name}/embeddings.py",
    f"{project_name}/vectorstore.py",
    f"{project_name}/retriever.py",
    f"{project_name}/chain.py",
    
    "notebooks/test.ipynb",
    
    "demo.py",
    "app.py",
    ".env",
    
    "requirements.txt",
    "setup.py",
    
    "static/css/style.css",
    "templates/index.html",
]

# Go to the list items and create's all the folder's and files
for filepath in list_of_files:
    
    filepath = Path(filepath)
    filedir , filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir , exist_ok = True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath , "w") as f:
            pass 
        
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")