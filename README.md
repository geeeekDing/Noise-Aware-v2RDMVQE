# Noise-Aware-v2RDMVQE

This repository contains the official source code, experimental data, and plotting scripts for our paper, "A Noise-Aware Framework for Obtaining Accurate Ground-State Properties on NISQ Devices".

Inside, you will find:
* The implementation of the core noise-aware framework presented in the paper.
* All the raw and processed data used to generate the figures in the publication.
* The scripts required to reproduce every figure from the paper.

---
## Overview

To fully reproduce all the results in this paper, two separate computational environments are required:

1.  **Data Generation Environment**: Used for classical computational chemistry simulations and generating Hamiltonians. This environment relies on an older version of Qiskit.
2.  **Analysis and Plotting Environment**: Used for running quantum computational chemistry calculations, SDP (Semidefinite Programming) calculations, and generating the final figures. This environment uses a newer version of Qiskit and its related libraries.

We strongly recommend using `conda` to create separate virtual environments to avoid package version conflicts.

---
## 1. Data Generation Environment Setup

This environment is used to generate the initial quantum chemistry computational data.

### Prerequisites:

* `pyscf==2.2.1`
* `openfermion`
* `openfermion-psi4==0.4`
* `qiskit==0.39.2`

### Installation (macOS / Linux):

```bash
# 1. Create a new conda environment
conda create -n Pyscf_VQE python==3.10

# 2. Activate the environment
conda activate Pyscf_VQE

# 3. Install the required packages
pip install pyscf==2.2.1
pip install openfermion
pip install openfermion-psi4==0.4
pip install qiskit==0.39.2
