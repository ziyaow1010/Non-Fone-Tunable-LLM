Code for arXiv:2602.00446 paper: Towards Building Non-Fine-Tunable Foundation Models

This repository contains the code implementation for the paper:
arXiv:2602.00446 https://arxiv.org/pdf/2602.00446

------------------------------------------------------------------------

Overview

This repository provides the training scripts and experimental setup
used in the paper arXiv:2602.00446.

The code includes: - Training scripts for different parameter freezing
settings - Configurations for DeepSpeed-based distributed training -
Scripts for experiments with frozen / partially frozen / non-finetunable
modules

The repository is intended for reproducing the experiments reported in
the paper.

------------------------------------------------------------------------

Repository Structure

. ├── train.py # Main training script ├── frozen.py # Training with
frozen components ├── frozen_un.py # Variant of frozen training
configuration ├── non-fine-tuable.py # Experiments with non-finetunable
modules │ ├── run.sh # Standard training launcher ├── run_frozen.sh #
Training launcher for frozen setting │ ├── deepspeed_config.json #
DeepSpeed configuration ├── requirements.txt # Python dependencies └──
.idea/ # IDE configuration

------------------------------------------------------------------------

File Description

train.py Main training script used for experiments in the paper.

frozen.py Implements the training pipeline where specific model
components are frozen during training.

frozen_un.py Variant of frozen training configuration used for
experiments with modified freezing strategies or batch configurations.

non-fine-tuable.py Implements experiments where certain modules are not
allowed to be fine-tuned.

run.sh Basic script to launch training. Example: bash run.sh

run_frozen.sh Launch script for experiments involving frozen components.
Example: bash run_frozen.sh

deepspeed_config.json Configuration file for DeepSpeed distributed
training, including: - optimizer settings - gradient accumulation -
memory optimization

requirements.txt Lists all Python dependencies required to run the code.
Install them with: pip install -r requirements.txt

------------------------------------------------------------------------

Installation

Clone the repository: git clone cd

Install dependencies: pip install -r requirements.txt

If using DeepSpeed: pip install deepspeed

------------------------------------------------------------------------

Training

Standard Training: bash run.sh

Frozen Training Experiments: bash run_frozen.sh

------------------------------------------------------------------------

Reproducing Paper Results

To reproduce the experiments reported in arXiv:2602.00446:

1.  Prepare the dataset described in the paper
2.  Install dependencies
3.  Run: bash run.sh

or

bash run_frozen.sh
