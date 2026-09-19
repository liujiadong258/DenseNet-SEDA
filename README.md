# DenseNet-SEDA: An Intelligent Model for Ore Image Recognition

## Purpose of the Code
This repository contains the source code for the manuscript DenseNet-SEDA: An Intelligent Model for Ore Image Recognition submitted to Computers & Geosciences. It provides the core PyTorch implementation of the DenseNet-SEDA model architecture, which integrates a Squeeze-and-Excitation attention mechanism and a dual-path classification framework.

## Hardware and Software Requirements
- Operating System: Windows
- Hardware: NVIDIA GPU with CUDA support
- Python: 3.12
- Dependencies: torch, torchvision, tqdm

## Repository Structure
- model.py: Defines the core DenseNet-SEDA neural network architecture.
- dataset.py: Contains the custom data loading pipeline.
- train.py: Script structure for model training.
- evaluate.py: Script structure for model evaluation.

## Usage and Path Configuration
Due to strict file size limits and data privacy, the original large-scale ore image dataset is not included in this repository.

To use this code with your own custom dataset, please place your image files locally and remember to open train.py and evaluate.py to manually modify the base_dir and photo_dir variables to match your actual absolute or relative file paths .

## Quick Verification
To quickly verify the structural integrity and functionality of the DenseNet-SEDA model without a dataset, you can directly execute the model script. It will instantiate the network and process a randomly generated dummy tensor to confirm the architecture is correctly defined:
python model.py
