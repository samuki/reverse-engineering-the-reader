# Reverse-Engineering the Reader

This repository contains the code for "Reverse-Engineering the Reader".

## Get Started

All experiments have been conducted using Python 3.10 using NVIDIA GeForce RTX 3090, RTX 4090 and RTX 2080 Ti GPUs.

### Install

Clone the repository and install the packages.

```
pip install -r requirements.txt
```

### Configuration

You can specify all configuration parameters for individual runs in `configs` and run a specific setting.
The main configuration files are in `configs/datasets_dll_wt`.

#### Logging
Note that the script will log metrics using Weights & Biases. To log the metrics, you'll need to add your API key in `model/utils/credentials.py`. To disable logging with wandb, set `use_wandb: False`.

#### Checkpoint Saving
To save checkpoints locally and not on the Euler cluster you need to specify `save_scratch: False`
If you are running scripts on Euler, and want to save them on the scratch space,  you need to add your username in `model/utils/credentials.py`

### Run Experiments

You can run an individual configuration:

```
python run.py --config_path /path/to/config
```