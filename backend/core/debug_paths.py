import os
import yaml

config_path = "configs/prod_params.yaml"
print(f"Current Working Directory: {os.getcwd()}")
print(f"Checking {config_path}: {os.path.exists(config_path)}")

# My robust logic
file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"File Directory: {file_dir}")
base_dir = os.path.abspath(os.path.join(file_dir, "..", ".."))
print(f"Base Directory (.. 2x): {base_dir}")
alt_path = os.path.join(base_dir, "configs", "prod_params.yaml")
print(f"Alt Path: {alt_path}")
print(f"Alt Path Exists: {os.path.exists(alt_path)}")

base_dir_3 = os.path.abspath(os.path.join(file_dir, "..", "..", ".."))
print(f"Base Directory (.. 3x): {base_dir_3}")
alt_path_3 = os.path.join(base_dir_3, "configs", "prod_params.yaml")
print(f"Alt Path 3 Exists: {os.path.exists(alt_path_3)}")
