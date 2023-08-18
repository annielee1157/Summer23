# Real-time Network Delay Monitor

This repository includes the following Plotly/DASH displays in the `apps` directory:
- `historical_view/real_time_net_delay_historical_view.py` monitors the cumulative arrival delay of the network by the date. The dates are selected from a time period of xxx and xxx in the past. The cumulative arrival delay is defined as the sum of all arrival delays at a given airport starting from 04:00:00 US/Eastern Time Zone. Then, the user may select the number of hours after 04:00:00 US/Eastern Time Zone to view the evolution of the cumulative delay

## Conda Environment

Start by setting up the project conda environment using conda.yaml:
```sh
conda env create -f conda.yml
```

Next, to activate the environment use: 
```sh
conda activate net-delay-monitor-env
```

To deactivate the environnment use: 
```sh
conda deactivate
```

If you need to update the environment at any time use:
```sh
conda env update --file conda.yml
```

**NOTE**
If using an M1/M2 Mac (i.e., a Mac with Apple Silicon), PyTorch Geometric is not available on Anaconda. Thus, it should be manually installed in the conda environment until this has been resolved. 

To do so:
1. Comment out the line `  - pyg=2.3.*` in `conda.yml` and create the conda environment using the command above. After the evnironment has been created, activate the environment.
2. If your [pip configuration file](https://pip.pypa.io/en/stable/topics/configuration/) has been modified to point to the NASA DIP version of PyPI, i.e., your `pip.conf` file looks like:

```
[global]  

index = http://localhost:8081/repository/dip-python/pypi

index-url = http://localhost:8081/repository/dip-python/simple

trusted-host = localhost
```
then, the following changes must be added (until the package is made available on the NASA DIP PyPI):
```
[global]  

index = http://localhost:8081/repository/dip-python/pypi

index-url = http://localhost:8081/repository/dip-python/simple

trusted-host = localhost
		pypi.org

extra-index-url = https://pypi.org/simple
```
3. `pip install torch_geometric`

*Future (when predictions are made available): The Python package `torch-geometric-temporal` may be used and can also only be installed via the general PyPI. As such, the changes to the pip configuration file (`pip.conf`) outlined above for M1/M2 Macs may be applied.*

## Parameters

config/parameters.yml has some variables that are used by multiple display pages.

## Setup Tunnels

To query data from the database, open the following tunnel:
```sh
ssh -fNL 5432:nas-warehouse:5432 username@lz101
```

<!-- To gather data for the STBO NEC airports, open the following tunnels:
```sh
ssh -fNL 5447:int1:5432 username@lz101
ssh -fNL 5448:int3:5432 username@lz101
ssh -fNL 5449:int10:5432 username@lz101
```
The machines correspond to the following airports: int1 is LGA, int3 is EWR, and int10 is JFK. -->

## Run the app

After setting up the needed tunnels, run the run.py script and open the corresponding port indicated in the terminal:
```sh
python run.py
```

## Data

The output data can be found in the corresponding folder for each display page.# Summer23
