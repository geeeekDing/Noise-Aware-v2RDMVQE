# Noise-Aware-v2RDMVQE

This repository contains the official source code, experimental data, and plotting scripts for our paper, "A Noise-Aware Framework for Obtaining Accurate Ground-State Properties on NISQ Devices".

Inside, you will find:

The implementation of the core noise-aware framework presented in the paper.

All the raw and processed data used to generate the figures in the publication.

The scripts required to reproduce every figure from the paper.

# Prerequisites

If you want to generate the classical computational chemistry results and Hamiltonians, you will need the following environment:

pyscf and openferimion

For quantum computational chemistry calculations, this project relies on qiskit. For SDP (Semidefinite Programming) calculations, it uses cvxpy.

# Installation

We strongly recommend creating a dedicated conda environment to ensure all dependencies are handled correctly.

Here is the recommended method for configuring your environment:

Bash

## 1. Create and activate a new conda environment named Qiskit1
conda create -n Qiskit1 python==3.12
conda activate Qiskit1

## 2. Install the required packages using pip
pip install qiskit==1.2
pip install qiskit-algorithms==0.3.0
pip install qiskit-aer==0.15.0
pip install joblib
pip install cvxpy
pip install scipy
pip install qiskit-nature
pip install pandas
pip install openfermion
