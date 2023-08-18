import yaml

from pathlib import Path

PARAMETER_FILE = 'parameters.yml'

with open(Path("config", PARAMETER_FILE)) as f:
    loaded_param_files = [PARAMETER_FILE]
    PARAMS = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Loaded parameter files: {loaded_param_files}")
    print(f"parameters: {PARAMS}")
