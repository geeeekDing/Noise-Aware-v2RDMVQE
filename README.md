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
pip install qiskit==0.39.2
pip install openfermion-psi4==0.4

pip install openfermion==1.7.1
pip install cmake (if need)
pip install pyscf==2.2.1
```

## 2. Analysis and Plotting Environment Setup

This environment is used to process the generated data, run SDP calculations, and plot all the figures in the paper.

### Prerequisites:

* `qiskit==1.2`
* `qiskit-algorithms`
* `qiskit-aer`
* `qiskit-nature`
* `cvxpy`

### Installation (macOS / Linux):

```bash
# 1. Create a new conda environment
conda create -n PaperAnalysis python==3.12

# 2. Activate the environment
conda activate PaperAnalysis

# 3. Install all required packages using pip
pip install qiskit==1.2
pip install qiskit-algorithms==0.3.0
pip install qiskit-aer==0.15.0
pip install qiskit-nature
pip install openfermion
pip install cvxpy
pip install joblib
pip install scipy
pip install pandas
```

## How to Reproduce

1.  First, activate the Data Generation Environment (conda activate Pyscf_VQE) and run the relevant scripts to generate the raw data.

2.  Next, activate the Analysis and Plotting Environment (conda activate PaperAnalysis) and run the analysis and plotting scripts to process the data and generate the final figures.

Happy reproducing! If you encounter any problems, please feel free to open an issue.
