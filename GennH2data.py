# ==============================================================================
# SCRIPT: VQE + VRDM for H2 Molecule
#
# Created Date: Sun Aug 31 2025
#
# Author: Qiming Ding, Huiyuan Wang, Yukun Zhang
#
# Description:
# This script integrates the VQE and VRDM calculation workflows for the H2
# molecule using a hardware-efficient ansatz (HEA) on a sto-3g basis.
#
# It first runs a VQE simulation (both noiseless and noisy) to compute the
# one- and two-body Reduced Density Matrices (RDMs). The noisy 2-RDM is then
# passed directly in-memory to the VRDM optimization routine.
# The VRDM routine finds the closest N-representable RDM to the noisy input
# and calculates the ground state energy.
#
#
# Copyright (c) 2025
# ==============================================================================

# Standard library imports
# from pathlib import Path
# import tqdm
# import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Pool
# import openfermion as of

# import mitiq
# from mitiq import cdr, Observable, PauliString
# from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
# from mitiq.interface.mitiq_qiskit import qiskit_utils

import time
import os
# Scientific computing and data handling libraries
import numpy as np
import pandas as pd
import pickle
import cvxpy as cp
from scipy.linalg import eigh
import datetime
import itertools
import functools
from functools import reduce, partial
from itertools import combinations

# np.set_printoptions(suppress=True, precision=6, threshold=np.inf)
from joblib import Memory, Parallel, delayed
from typing import Union, List, Optional, Tuple, Any
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.quantum_info import Clifford
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, SGate, EfficientSU2,SdgGate, IGate


from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_aer import AerSimulator, AerError
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.primitives import EstimatorV2 
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import L_BFGS_B,COBYLA
from qiskit_algorithms import VQE
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient,ParamShiftEstimatorGradient

from qiskit.exceptions import QiskitError



def measure_execution_time(func):
    """
    Decorator function to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result

    return wrapper


# ==============================================================================

# Part A

def to_density_matrix(ground_state_wavefunction):
    """
    Convert a given quantum state (either a state vector or a density matrix) 
    into its density matrix representation.

    Parameters:
        ground_state_wavefunction (numpy.ndarray): The quantum state to be converted.
            It can be provided as a state vector (1D array) or already as a density 
            matrix (2D square matrix).

    Returns:
        numpy.ndarray: The density matrix representation of the input state.
            If the input is a state vector, it returns the corresponding density matrix.
            If the input is already a density matrix, it returns it unchanged.

    Raises:
        ValueError: If the input is not a valid quantum state, i.e., it's not a 
            1D vector (state vector) or a 2D square matrix (density matrix).
    """
    # Check if the input is a 1D array (pure state vector)
    if ground_state_wavefunction.ndim == 1:
        # If it's a state vector, convert it to a density matrix using the outer product
        # ρ = |ψ⟩⟨ψ| where ψ is the state vector
        density_matrix = np.outer(ground_state_wavefunction, np.conj(ground_state_wavefunction))
        return density_matrix
    
    # Check if the input is already a 2D array (potential density matrix)
    elif ground_state_wavefunction.ndim == 2:
        # Ensure it's a square matrix (valid density matrix must be square)
        if ground_state_wavefunction.shape[0] != ground_state_wavefunction.shape[1]:
            raise ValueError("Input must be a square matrix to represent a valid density matrix.")
        # If it's a valid square matrix, return it directly
        return ground_state_wavefunction
    
    else:
        # Raise an error if the input is neither a 1D vector nor a 2D square matrix
        raise ValueError("Input must be either a 1D state vector or a 2D square matrix.")
    
def calculate_frobenius_norm_difference(theoretical_data, experimental_data):
    """
    Calculate the Frobenius norm difference between theoretical and experimental data.
    This function computes the Frobenius norm, which measures the overall difference 
    between two arrays (matrices or tensors), as well as returning the absolute 
    element-wise differences between the input arrays.

    Parameters:
        theoretical_data (numpy.ndarray): Theoretical data represented as a Numpy array, 
                                          can be a matrix (2D) or tensor (4D).
        experimental_data (numpy.ndarray): Experimental data represented as a Numpy array, 
                                           must have the same shape as theoretical_data.

    Returns:
        float: The Frobenius norm difference, representing the overall magnitude of the 
               difference between the two input arrays.
        numpy.ndarray: A tensor of the absolute element-wise differences between 
                      theoretical_data and experimental_data.

    Raises:
        ValueError: If the input arrays do not have the same shape or the dimensionality
                    is neither 2D nor 4D.
    """
    # Ensure both input arrays have the same shape
    if theoretical_data.shape != experimental_data.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Handle 2D (matrix) or 4D (tensor) input arrays
    if theoretical_data.ndim == 2:
        # If the input is a 2D matrix, calculate the Frobenius norm directly using np.linalg.norm
        norm_difference = np.linalg.norm(theoretical_data - experimental_data, 'fro')
    elif theoretical_data.ndim == 4:
        # If the input is a 4D tensor, manually calculate the Frobenius norm

        # Step 1: Compute the difference between theoretical and experimental tensors
        difference = theoretical_data - experimental_data
        
        # Step 2: Square each element of the difference tensor
        squared_difference = np.square(difference)
        
        # Step 3: Sum all the squared elements to get the sum of squares
        sum_of_squares = np.sum(squared_difference)
        
        # Step 4: Take the square root of the sum of squares to obtain the Frobenius norm
        norm_difference = np.sqrt(sum_of_squares)

        # Print intermediate results for comparison of different methods (commented out)
        # Uncomment if you want to compare various methods of calculating the Frobenius norm
        '''
        # Method 2: Directly using numpy's built-in norm function
        norm_difference2 = np.linalg.norm(theoretical_data - experimental_data)
        print("Method 2 - numpy.linalg.norm on difference:", norm_difference2)

        # Method 3: Using numpy.einsum to efficiently compute the Frobenius norm
        norm_difference3 = np.sqrt(np.einsum('ijkl,ijkl', difference, difference))
        print("Method 3 - numpy.einsum:", norm_difference3)

        # Method 4: Flatten the tensor and calculate the norm on the flattened array
        norm_difference4 = np.linalg.norm(difference.ravel())
        print("Method 4 - Flattening and numpy.linalg.norm:", norm_difference4)
        '''
    else:
        # Raise an error if the input is neither a 2D matrix nor a 4D tensor
        raise ValueError("Input arrays must be either two-dimensional (matrix) or four-dimensional (tensor).")

    # Calculate the absolute element-wise difference tensor
    # This represents the magnitude of the difference at each corresponding element
    element_wise_difference = np.abs(theoretical_data - experimental_data)

    # Return the Frobenius norm and the element-wise difference tensor
    return norm_difference.real, element_wise_difference


def noisy_model_set(prob_1, prob_2):
    """
    Create a noise model with depolarizing errors for 1-qubit and 2-qubit gates.
    If both prob_1 and prob_2 are zero, the noise model is set to None.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.

    Returns:
        NoiseModel or None: The noise model with the specified depolarizing errors.
    """
    if prob_1 == 0 and prob_2 == 0:
        noise_model = None
    else:
        error_1 = depolarizing_error(prob_1, 1)
        error_2 = depolarizing_error(prob_2, 2)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error_1, 
            ['u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg']
        )    
        noise_model.add_all_qubit_quantum_error(
            error_2, 
            ['swap', 'cx', 'cy', 'cz', 'csx', 'cp', 'cu', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy', 'rzz', 'rzx', 'ecr']
        )
    return noise_model

def build_noisy_estimator(prob_1, prob_2, shots_num, device="CPU"):
    """
    Build a noisy quantum estimator using the specified noise model, and attempt
    to use GPU if the 'device' parameter is set to 'GPU'. If GPU initialization fails,
    fallback to CPU.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.
        shots_num (int): Number of shots to use for simulation.
        device (str): 'GPU' or 'CPU'. If 'GPU' is selected, the code will attempt to 
                      initialize the estimator on GPU.

    Returns:
        Tuple: (noise_model, noisy_estimator)
    """
    noise_model = None
    noisy_estimator = None

    if prob_1 != 0 or prob_2 != 0:
        # Create the noise model with the specified error probabilities
        noise_model = noisy_model_set(prob_1, prob_2)
        try:
            # Attempt to initialize the AerEstimator with GPU if specified
            if device.upper() == "GPU":
                print("Attempting to initialize GPU for noisy estimator...")
                noisy_estimator = AerEstimator(
                    backend_options={"method": "density_matrix", "device": "GPU", "noise_model": noise_model},
                    run_options={"shots": shots_num}
                )
            else:
                # Use CPU by default if no GPU is requested
                print("Using CPU for noisy estimator...")
                noisy_estimator = AerEstimator(
                    backend_options={"method": "density_matrix", "noise_model": noise_model},
                    run_options={"shots": shots_num}
                )
        except AerError as e:
            # If GPU initialization fails, fallback to CPU and log the error
            print("Failed to initialize GPU estimator, falling back to CPU:", str(e))
            noisy_estimator = AerEstimator(
                backend_options={"method": "density_matrix", "noise_model": noise_model},
                run_options={"shots": shots_num}
            )

    return noise_model, noisy_estimator
'''
def build_noisy_estimatorV2(prob_1, prob_2, shots_num, device="CPU"):
    """
    Build a noisy quantum estimator using EstimatorV2, and attempt to use GPU 
    if the 'device' parameter is set to 'GPU'. If GPU initialization fails, fallback to CPU.

    Parameters:
        prob_1 (float): Depolarizing error probability for 1-qubit gates.
        prob_2 (float): Depolarizing error probability for 2-qubit gates.
        shots_num (int): Number of shots to use for simulation.
        device (str): 'GPU' or 'CPU'. If 'GPU' is selected, the code will attempt to 
                      initialize the estimator on GPU.

    Returns:
        Tuple: (noise_model, noisy_estimator)
    """
    if prob_1 != 0 or prob_2 != 0:
        # Create the noise model with the specified error probabilities
        noise_model = noisy_model_set(prob_1, prob_2)

        # Set the backend device (GPU or CPU)
        method = "density_matrix"
        backend_device = "GPU" if device.upper() == "GPU" else "CPU"

        # Initialize the simulator with noise model
        try:
            if backend_device == "GPU":
                print("Attempting to initialize GPU for noisy estimator...")
                simulator = AerSimulator(method=method, device="GPU", noise_model=noise_model)
            else:
                print("Using CPU for noisy estimator...")
                simulator = AerSimulator(method=method, noise_model=noise_model)

            # Create EstimatorV2 using the simulator as the backend
            noisy_estimatorV2 = EstimatorV2.from_backend(
                backend=simulator,
                options={
                    "backend_options": {"noise_model": noise_model},
                    "run_options": {"shots": shots_num},
                    "default_precision": 0.01  # Adjust precision as needed
                }
            )
        except AerError as e:
            # If GPU initialization fails, fallback to CPU and log the error
            print("Failed to initialize GPU estimator, falling back to CPU:", str(e))
            simulator = AerSimulator(method=method, noise_model=noise_model)
            noisy_estimatorV2 = EstimatorV2.from_backend(
                backend=simulator,
                options={
                    "backend_options": {"noise_model": noise_model},
                    "run_options": {"shots": shots_num},
                    "default_precision": 0.01
                }
            )

    return noisy_estimatorV2
'''
@measure_execution_time
def Gene_Qiskit_VQE_hamiltonian(n_qubits, qubit_hamiltonian):
    """
    Generate a Qiskit-compatible Hamiltonian for VQE (Variational Quantum Eigensolver)
    from a given qubit Hamiltonian.

    Parameters:
        n_qubits (int): The number of qubits in the system.
        qubit_hamiltonian (QubitOperator): The qubit Hamiltonian in terms of Pauli operators.
    
    Returns:
        SparsePauliOp: A Qiskit SparsePauliOp object representing the Hamiltonian.

    Description:
        The function converts a Hamiltonian, represented as a dictionary of Pauli terms 
        and coefficients, into a Qiskit SparsePauliOp, which can be used for quantum 
        simulations and optimizations like VQE. The qubit Hamiltonian is expected to 
        have terms of Pauli strings (e.g., X, Y, Z), and each term is translated into 
        a corresponding Pauli string operator in Qiskit.

    Example:
        A Hamiltonian term like 'X0 Y1 Z2' is converted into a SparsePauliOp representation.

    """
    paulis = []  # List to store the Pauli strings for each term
    coeffs = []  # List to store the coefficients for each Pauli string

    # Loop through each term in the qubit Hamiltonian (terms are Pauli operators and coefficients)
    for term, coeff in qubit_hamiltonian.terms.items():
        modes = [0] * n_qubits  # Initialize a list to store Pauli modes for each qubit (I, X, Y, Z)

        # For each Pauli operator in the term, assign its corresponding integer mode
        for qubit, op in term:
            if op == "X":
                modes[qubit] = 1  # X Pauli operator
            elif op == "Y":
                modes[qubit] = 2  # Y Pauli operator
            elif op == "Z":
                modes[qubit] = 3  # Z Pauli operator

        # Construct the Pauli string from the list of modes, in reversed order for correct indexing
        pauli_str = "".join(["I", "X", "Y", "Z"][mode] for mode in reversed(modes))

        # Append the Pauli string and the corresponding coefficient to their respective lists
        paulis.append(pauli_str)
        coeffs.append(coeff)

    # Combine Pauli strings and their coefficients into a list of tuples
    pauli_op = [(pauli, weight) for pauli, weight in zip(paulis, coeffs)]

    # Convert the list of Pauli terms into a SparsePauliOp (Qiskit representation)
    hamiltonian_qiskit = SparsePauliOp.from_list(pauli_op)

    # Print the number of qubits for reference/debugging
    print(f"Number of qubits: {hamiltonian_qiskit.num_qubits}")

    return hamiltonian_qiskit  # Return the constructed Qiskit Hamiltonian

def run_vqe(hamiltonian_qiskit, ansatz, optimizer, estimator):
    """
    Generic function to run the Variational Quantum Eigensolver (VQE) algorithm.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format (e.g., SparsePauliOp).
    
    ansatz : QuantumCircuit
        The ansatz quantum circuit used for the VQE algorithm.
    
    optimizer : Optimizer
        The classical optimizer used to minimize the expectation value of the Hamiltonian.
    
    estimator : Estimator
        The estimator object used to calculate the expectation values of the Hamiltonian 
        for given parameters (supports both noiseless and noisy estimators).

    Returns:
    --------
    result : VQEResult
        The result object containing details of the VQE optimization process (e.g., optimal energy).
    
    parameters_list : list
        A list of parameter values collected during the VQE optimization process.
    
    circ : QuantumCircuit
        The optimal circuit after the VQE optimization with parameters assigned.
    
    estimator : Estimator
        The estimator used in the VQE, returned for further reuse or analysis.
    
    optimal_parameters : ndarray
        The optimal parameters obtained from the VQE optimization process.
    """
    # Store intermediate results such as iteration count, parameter values, and energy values
    counts = []
    values = []
    parameters_list = []
    std_list = []

    # Callback function to store intermediate VQE results
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        parameters_list.append(parameters)
        std_list.append(std)
        # Uncomment the following line for more detailed output during optimization
        # print(f"Iteration {eval_count}: Energy = {mean:.8f}, Parameters = {parameters}")

    # Using the ParamShiftEstimatorGradient for gradient calculation (parameter shift rule)
    gradient = ParamShiftEstimatorGradient(estimator)
    
    # Initialize the VQE algorithm with estimator, ansatz, optimizer, and gradient
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, gradient=gradient, callback=store_intermediate_result)
    
    # Set the initial parameters for the ansatz (either zeros or random values)
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    # Alternatively, you could use random initial points:
    # vqe.initial_point = np.random.uniform(low=-np.pi, high=np.pi, size=ansatz.num_parameters)

    # Run the VQE algorithm to find the minimum eigenvalue of the Hamiltonian
    result = vqe.compute_minimum_eigenvalue(hamiltonian_qiskit)

    # Get the optimal quantum circuit with the best-found parameters
    circ = result.optimal_circuit.assign_parameters(result.optimal_parameters)
    optimal_parameters = result.optimal_parameters

    # Print the final optimal energy found by the VQE
    print(f"VQE Optimal Energy: {result.optimal_value.real:.8f}")

    return result, parameters_list, circ, estimator, optimal_parameters

def create_ansatz(num_qubits, n_particles):
    """
    Create the Hartree-Fock initial state and UCCSD (Unitary Coupled Cluster with Single and Double excitations) ansatz.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits in the quantum system, which corresponds to the number of orbitals in the problem.
    
    n_particles : int
        The total number of particles (electrons) in the quantum system.

    Returns:
    --------
    ansatz : UCCSD
        The UCCSD ansatz initialized with the Hartree-Fock state as the reference state.
    """
    # Initialize the qubit mapper (Jordan-Wigner mapping) for mapping fermions to qubits
    mapper = JordanWignerMapper()
    
    # Number of particles split between spin-up and spin-down electrons
    num_particles = [n_particles // 2, n_particles // 2]
    
    # Create the Hartree-Fock initial state as the starting point for the ansatz
    hf = HartreeFock(num_spatial_orbitals=num_qubits // 2, 
                     num_particles=num_particles, 
                     qubit_mapper=mapper)
    
    # Define the UCCSD ansatz with the Hartree-Fock state as the reference
    ansatz = UCCSD(num_spatial_orbitals=num_qubits // 2, 
                   num_particles=num_particles, 
                   qubit_mapper=mapper,
                   initial_state=hf, 
                   generalized=False,  # Set to True if you want to use generalized excitations
                   preserve_spin=True)  # Preserve the total spin symmetry in the system

    return ansatz

def create_hardware_efficient_ansatz(num_qubits: int, 
                                     n_particles: int, 
                                     reps: int = 1, 
                                     entanglement: str = 'linear'):
    """
    创建一个硬件高效的 ansatz (EfficientSU2)，并使用 Hartree-Fock 态作为初始态。
    ... (函数体与之前相同)
    """
    
    mapper = JordanWignerMapper()
    num_particles_tuple = (n_particles // 2, n_particles // 2)
    
    hf_initial_state = HartreeFock(num_spatial_orbitals=num_qubits // 2,
                                   num_particles=num_particles_tuple,
                                   qubit_mapper=mapper)
    
    ansatz = EfficientSU2(num_qubits=num_qubits,
                          su2_gates=['rx', 'y'],
                          reps=reps,
                          entanglement=entanglement,
                          initial_state=hf_initial_state)

    return ansatz

def HEA_VQE_noiseless(hamiltonian_qiskit, num_qubits, n_particles, HEAp, max_iterations=1000):
    """
    Runs noiseless HEA VQE.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format.
    
    num_qubits : int
        The number of qubits in the system.
    
    n_particles : int
        The number of particles in the system.
    
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 1000).

    Returns:
    --------
    result : VQEResult
        The result object from the VQE algorithm.
    """
    # Create the ansatz (HEA with Hartree-Fock initial state)
    # ansatz = create_ansatz(num_qubits, n_particles)
    ansatz = create_hardware_efficient_ansatz(num_qubits, n_particles,HEAp)
    # Define the classical optimizer (COBYLA in this case)
    optimizer = COBYLA(maxiter=max_iterations)
    
    # Use the default noiseless estimator
    estimator = Estimator()
    # estimator = StatevectorEstimator()

    # Run the VQE algorithm using the provided Hamiltonian, ansatz, optimizer, and estimator
    return run_vqe(hamiltonian_qiskit, ansatz, optimizer, estimator)

def HEA_VQE_noisy(hamiltonian_qiskit, num_qubits, n_particles, noisy_estimator, HEAp, max_iterations=1000):
    """
    Runs noisy HEA VQE with the specified noise model.

    Parameters:
    -----------
    hamiltonian_qiskit : OperatorBase
        The Hamiltonian of the system in Qiskit's format.
    
    num_qubits : int
        The number of qubits in the system.
    
    n_particles : int
        The number of particles in the system.
    
    noisy_estimator : Estimator
        A noisy estimator created externally and passed into the function.
    
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 1000).

    Returns:
    --------
    result : VQEResult
        The result object from the noisy VQE algorithm.
    """
    # Create the ansatz (HEA with Hartree-Fock initial state)
    # ansatz = create_ansatz(num_qubits, n_particles)
    # Create the ansatz (HEA with Hartree-Fock initial state)
    ansatz = create_hardware_efficient_ansatz(num_qubits, n_particles)
    # Define the classical optimizer (COBYLA in this case)
    optimizer = COBYLA(maxiter=max_iterations)

    # Run the VQE algorithm using the provided Hamiltonian, ansatz, optimizer, and noisy estimator
    return run_vqe(hamiltonian_qiskit, ansatz, optimizer, noisy_estimator)

def get_one_rdm_reduced_indices(num_qubits: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_qubits) for j in range(i, num_qubits)]

def get_one_rdm_term(i, j, num_qubits, circ, estimator):
    """
    Calculate a specific element of the one-body reduced density matrix (1-RDM)
    for the given indices in a quantum system.

    Parameters:
    -----------
    i, j : int
        Indices for the 1-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (i, j, float)
        A tuple containing the indices (i, j) and the calculated value of the 1-RDM element.
    """
    # Check if the indices are out of bounds for the system size
    if i >= num_qubits or j >= num_qubits:
        # Return 0 if indices are invalid
        return i, j, 0.0
    
    # Create the fermionic operator for the 1-RDM element (creation at i, annihilation at j)
    fermi_term = FermionicOp({f"+_{i} -_{j}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize the total expectation value
    total_expectation_value = 0.0
    
    # Loop over Pauli terms in the mapped qubit operator
    for pauli_op, coeff in qubit_term.label_iter():
        # Create a SparsePauliOp for each Pauli term
        single_pauli_term = SparsePauliOp(pauli_op)
        
        # Use the estimator to compute the expectation value of the Pauli term
        result = estimator.run(circuits=[circ], observables=[single_pauli_term]).result()
        
        # Multiply the expectation value by the coefficient and accumulate it
        expectation_value = np.real(coeff * result.values[0])
        total_expectation_value += expectation_value

    # Output the calculated 1-RDM element for debugging
    # print(f"Calculated 1-RDM element ({i}, {j}) is {total_expectation_value}")
    
    # Return the indices and the calculated value
    return i, j, total_expectation_value

@measure_execution_time
def get_one_rdm(num_qubits, circ=None, estimator=None):
    """
    Calculate the full one-body reduced density matrix (1-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    one_rdm : np.ndarray
        A 2D numpy array representing the one-body reduced density matrix.
    """
    # Ensure that an estimator is provided for the calculation
    if estimator is None:
        raise ValueError("Estimator must be provided for 1-RDM calculation.")
        
    # Initialize an empty matrix for the 1-RDM
    one_rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    
    # Partial application of the function to fix the constant parameters
    func = functools.partial(get_one_rdm_term, num_qubits=num_qubits, circ=circ, estimator=estimator)
    
    # Generate all possible index combinations (i, j) for the 1-RDM
    indices = [(i, j) for i in range(num_qubits) for j in range(num_qubits)]
    
    # Parallel computation of all 1-RDM elements (set n_jobs according to your system)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j) for i, j in indices)

    # Fill the 1-RDM matrix with the computed results
    for result in results:
        i, j, temp = result
        one_rdm[i, j] = temp
        one_rdm[j, i] = temp
    return one_rdm

def is_invalid_index(i: int, j: int, k: int, l: int, num_qubits: int) -> bool:
    """
    Check if the given indices are invalid based on the number of spin orbitals.

    Args:
        i: The first index.
        j: The second index.
        k: The third index.
        l: The fourth index.
        num_qubits: The total number of spin orbitals.

    Returns:
        True if the indices are invalid, False otherwise.
    """
    valid_term = (
        (i < num_qubits // 2 and j < num_qubits // 2 and k < num_qubits // 2 and l < num_qubits // 2) or
        (i < num_qubits // 2 and j >= num_qubits // 2 and k < num_qubits // 2 and l >= num_qubits // 2) or
        (i < num_qubits // 2 and j >= num_qubits // 2 and k >= num_qubits // 2 and l < num_qubits // 2) or
        (i >= num_qubits // 2 and j < num_qubits // 2 and k < num_qubits // 2 and l >= num_qubits // 2) or
        (i >= num_qubits // 2 and j < num_qubits // 2 and k >= num_qubits // 2 and l < num_qubits // 2) or
        (i >= num_qubits // 2 and j >= num_qubits // 2 and k >= num_qubits // 2 and l >= num_qubits // 2)
    )
    return not valid_term

def get_two_rdm_reduced_indices(num_qubits: int) -> List[Tuple[int, int, int, int]]:
    unique_indices = []
    for i, j, k, l in itertools.product(range(num_qubits), repeat=4):
        if (i, j, k, l) not in unique_indices and (k, l, i, j) not in unique_indices and \
           (j, i, k, l) not in unique_indices and (i, j, l, k) not in unique_indices:
            unique_indices.append((i, j, k, l))
    return unique_indices

def get_total_two_rdm_terms(num_qubits: int) -> int:
    """
    Calculate the total number of RDM terms to be computed.

    Args:
        num_qubits: The total number of spin orbitals.
        num_particles: The total number of particles.

    Returns:
        The total number of RDM terms.
    """
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    valid_terms = 0

    for i, j, k, l in unique_indices:
        if not is_invalid_index(i, j, k, l, num_qubits):
            valid_terms += 1

    return valid_terms

def get_two_rdm_term(i, j, k, l, num_qubits, circ, estimator):
    """
    Calculate a specific element of the two-body reduced density matrix (2-RDM) 
    for given indices in a quantum system.

    Parameters:
    -----------
    i, j, k, l : int
        Indices for the 2-RDM element.
    
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    circ : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (i, j, k, l, float)
        A tuple containing the indices (i, j, k, l) and the calculated value of the 2-RDM element.
    """
    # Check if the given indices are invalid for the current system size
    if is_invalid_index(i, j, k, l, num_qubits):
        # If invalid, return the indices and a value of 0.0
        return i, j, k, l, 0.0
    
    # Construct the fermionic operator for the 2-RDM element
    # Fermionic operator format: f"+_{i} +_{j} -_{k} -_{l}"
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

    # Use Jordan-Wigner mapping to convert the fermionic operator to a qubit operator
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize total expectation value to accumulate the results
    total_expectation_value = 0.0

    # Loop over Pauli terms in the qubit operator
    for pauli_op, coeff in sorted(qubit_term.label_iter()):
        # Create a SparsePauliOp for each individual Pauli term
        single_pauli_term = SparsePauliOp(pauli_op, coeffs=1)
        
        # Use the estimator to compute the expectation value of the Pauli term
        result = estimator.run(circuits=[circ], observables=[single_pauli_term]).result()
        
        # Multiply the expectation value by the coefficient and accumulate it
        expectation_value = np.real(coeff * result.values[0])
        total_expectation_value += expectation_value

    # Store the final calculated value of the 2-RDM element
    temp = total_expectation_value
    
    # Output the calculated 2-RDM element for debugging purposes
    # print(f"Calculated 2-RDM element ({i}, {j}, {k}, {l}) is {temp}")
    # Return the indices and the calculated 2-RDM element
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    # Use functools.partial to fix the constant parameters (num_qubits, circ, estimator)
    func = functools.partial(get_two_rdm_term, num_qubits=num_qubits, circ=circ, estimator=estimator)

    # Use joblib's Parallel to compute 2-RDM elements in parallel (can adjust n_jobs for parallelism)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the calculated values, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm


def generate_pauli_terms(unique_indices, num_qubits):
    """
    生成所有需要的 Pauli 算符及其系数，并建立 RDM 元素与 Pauli 项的映射。

    Parameters:
    -----------
    unique_indices : list of tuples
        需要计算的 RDM 索引组合。

    num_qubits : int
        系统中的量子比特数量。

    Returns:
    --------
    pauli_dict : dict
        Pauli 算符及其累积系数的字典。

    rdm_pauli_terms : dict
        RDM 元素与对应的 Pauli 项及系数的映射。

    """
    pauli_dict = {}
    rdm_pauli_terms = {}
    mapper = JordanWignerMapper()

    for idx in unique_indices:
        i, j, k, l = idx

        # 检查索引的有效性
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # 构建费米子算符
        fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

        # 映射到量子比特算符
        qubit_term = mapper.map(fermi_term)

        # 遍历量子比特算符的 Pauli 项
        for pauli_op, coeff in qubit_term.to_list():
            # 更新 Pauli 项的系数
            if pauli_op in pauli_dict:
                pauli_dict[pauli_op] += coeff
            else:
                pauli_dict[pauli_op] = coeff

            # 记录 RDM 元素与 Pauli 项的关系
            if idx in rdm_pauli_terms:
                rdm_pauli_terms[idx].append((pauli_op, coeff))
            else:
                rdm_pauli_terms[idx] = [(pauli_op, coeff)]

    return pauli_dict, rdm_pauli_terms

def compute_expectation_values(pauli_dict, circ, estimator):
    """
    并行计算所有唯一 Pauli 算符的期望值。

    Parameters:
    -----------
    pauli_dict : dict
        Pauli 算符及其累积系数的字典。

    circ : QuantumCircuit
        用于制备量子态的量子电路。

    estimator : Estimator
        用于计算期望值的估计器对象。

    Returns:
    --------
    pauli_expectation_dict : dict
        Pauli 算符及其期望值的字典。

    """
    # 提取所有唯一的 Pauli 算符
    unique_pauli_ops = list(pauli_dict.keys())

    # 将 Pauli 算符转换为 SparsePauliOp 对象
    sparse_pauli_ops = [SparsePauliOp(pauli_op) for pauli_op in unique_pauli_ops]

    # 定义计算单个 Pauli 项期望值的函数
    def compute_expectation(pauli_op):
        result = estimator.run(circuits=[circ], observables=[pauli_op]).result()
        return result.values[0]

    # 并行计算期望值
    expectation_values = Parallel(n_jobs=-1)(
        delayed(compute_expectation)(pauli_op) for pauli_op in sparse_pauli_ops
    )

    # 构建 Pauli 算符与期望值的映射
    pauli_expectation_dict = {
        pauli_op: value for pauli_op, value in zip(unique_pauli_ops, expectation_values)
    }

    return pauli_expectation_dict

def get_two_rdm_term_Parallel(i, j, k, l, num_qubits,pauli_expectation_dict, circ, estimator):
        # Check if the given indices are invalid for the current system size
    if is_invalid_index(i, j, k, l, num_qubits):
        # If invalid, return the indices and a value of 0.0
        return i, j, k, l, 0.0
    
    # Construct the fermionic operator for the 2-RDM element
    # Fermionic operator format: f"+_{i} +_{j} -_{k} -_{l}"
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

    # Use Jordan-Wigner mapping to convert the fermionic operator to a qubit operator
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Initialize total expectation value to accumulate the results
    total_expectation_value = 0.0

    # Loop over Pauli terms in the qubit operator
    for pauli_op, coeff in sorted(qubit_term.label_iter()):
        expectation_value = coeff * pauli_expectation_dict[pauli_op]
        total_expectation_value += expectation_value

    # Store the final calculated value of the 2-RDM element
    temp = np.real(total_expectation_value)
    
    # Output the calculated 2-RDM element for debugging purposes
    # print(f"Calculated 2-RDM element with Parallel after GPU Parallel 3th way, ({i}, {j}, {k}, {l}) is {temp}")
    # Return the indices and the calculated 2-RDM element
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm_Parallel3(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.
    
    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)

    # 获取需要计算的 RDM 索引组合
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # 生成 Pauli 项和对应关系
    pauli_dict, rdm_pauli_terms = generate_pauli_terms(unique_indices, num_qubits)
    # print(pauli_dict)
    # print(rdm_pauli_terms)

    # 计算 Pauli 项的期望值
    pauli_expectation_dict = compute_expectation_values(pauli_dict, circ, estimator)
    
    print(pauli_expectation_dict)
    
    # Use functools.partial to fix the constant parameters (num_qubits, circ, estimator)
    func = functools.partial(get_two_rdm_term_Parallel, num_qubits=num_qubits, pauli_expectation_dict = pauli_expectation_dict, circ=circ, estimator=estimator)

    # Use joblib's Parallel to compute 2-RDM elements in parallel (can adjust n_jobs for parallelism)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the calculated values, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm

def assemble_two_rdm(unique_indices, num_qubits, pauli_expectation_dict, rdm_pauli_terms):
    """
    使用期望值和预处理信息，组装完整的 2-RDM。

    Parameters:
    -----------
    unique_indices : list of tuples
        需要计算的 RDM 索引组合。

    num_qubits : int
        系统中的量子比特数量。

    pauli_expectation_dict : dict
        Pauli 算符及其期望值的字典。

    rdm_pauli_terms : dict
        RDM 元素与对应的 Pauli 项及系数的映射。

    Returns:
    --------
    two_rdm : np.ndarray
        计算得到的二体约化密度矩阵。

    """    
    # 初始化 2-RDM 矩阵
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # 遍历所有 RDM 元素
    for idx in unique_indices:
        i, j, k, l = idx

        # 跳过无效索引
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # 初始化当前 RDM 元素的总期望值
        total_expectation_value = 0.0

        # 获取对应的 Pauli 项和系数
        pauli_terms = rdm_pauli_terms.get(idx, [])

        # 计算总期望值
        for pauli_op, coeff in pauli_terms:
            expectation_value = coeff * pauli_expectation_dict[pauli_op]
            total_expectation_value += expectation_value

        # 获取实部
        temp = np.real(total_expectation_value)
        
        # Output the calculated 2-RDM element for debugging purposes
        # print(f"Calculated 2-RDM element with Parallel after GPU Parallel 2nd way, ({i}, {j}, {k}, {l}) is {temp}")
        
        # 填充 2-RDM 矩阵，考虑对称性和反对称性
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # 对称性
        two_rdm[j, i, k, l] = -temp      # 在 i, j 索引上的反对称性
        two_rdm[i, j, l, k] = -temp      # 在 k, l 索引上的反对称性

    return two_rdm

@measure_execution_time
def get_two_rdm_Parallel2(num_qubits, circ=None, estimator=None):
    """
    计算量子系统的二体约化密度矩阵（2-RDM）。

    Parameters:
    -----------
    num_qubits : int
        系统中的量子比特数量。

    circ : QuantumCircuit, optional
        用于制备量子态的量子电路。

    estimator : Estimator
        用于计算期望值的估计器对象。

    Returns:
    --------
    two_rdm : np.ndarray
        二体约化密度矩阵。

    """
    # 确保提供了估计器
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")

    # 获取需要计算的 RDM 索引组合
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # 生成 Pauli 项和对应关系
    pauli_dict, rdm_pauli_terms = generate_pauli_terms(unique_indices, num_qubits)
    
    # print(pauli_dict)
    
    # print(rdm_pauli_terms)

    # 计算 Pauli 项的期望值
    pauli_expectation_dict = compute_expectation_values(pauli_dict, circ, estimator)
    
    # print(pauli_expectation_dict)
    
    # 组装 2-RDM
    
    two_rdm = assemble_two_rdm(unique_indices, num_qubits, pauli_expectation_dict, rdm_pauli_terms)

    return two_rdm

@measure_execution_time
def get_two_rdm_Parallel(num_qubits, circ=None, estimator=None):
    """
    Calculate the full two-body reduced density matrix (2-RDM) for a quantum system.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.

    circ : QuantumCircuit, optional
        The quantum circuit used to prepare the quantum state. Must be provided.

    estimator : Estimator
        The estimator object used to compute expectation values (required).

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure an estimator is provided, as it's necessary for the calculation
    if estimator is None:
        raise ValueError("An estimator must be provided for 2-RDM calculation.")

    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)

    # Initialize a dictionary to store Pauli terms and their total coefficients
    pauli_dict = {}

    # Initialize a dictionary to map RDM elements to their corresponding Pauli terms and coefficients
    rdm_pauli_terms = {}

    mapper = JordanWignerMapper()

    # Preprocessing: Generate Pauli terms for all RDM elements
    for idx in unique_indices:
        i, j, k, l = idx

        # Check if the given indices are invalid for the current system size
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # Construct the fermionic operator for the 2-RDM element
        fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)

        # Map the fermionic operator to a qubit operator
        qubit_term = mapper.map(fermi_term)

        # For each Pauli term in the qubit operator, accumulate its coefficient
        for pauli_op, coeff in qubit_term.to_list():
            if pauli_op in pauli_dict:
                pauli_dict[pauli_op] += coeff
            else:
                pauli_dict[pauli_op] = coeff

            # Map RDM elements to their corresponding Pauli terms and coefficients
            if idx in rdm_pauli_terms:
                rdm_pauli_terms[idx].append((pauli_op, coeff))
            else:
                rdm_pauli_terms[idx] = [(pauli_op, coeff)]

    # Get the unique Pauli operators
    unique_pauli_ops = list(pauli_dict.keys())
    # print(unique_pauli_ops)
    # Convert Pauli operators to SparsePauliOp
    sparse_pauli_ops = [SparsePauliOp(pauli_op) for pauli_op in unique_pauli_ops]
    # print(sparse_pauli_ops)
    # Create a list of coefficients corresponding to the Pauli operators
    pauli_coeffs = [pauli_dict[pauli_op] for pauli_op in unique_pauli_ops]
    # print(pauli_coeffs)
    
    # Parallel computation of expectation values
    def compute_expectation(pauli_op):
        result = estimator.run(circuits=[circ], observables=[pauli_op]).result()
        return result.values[0]

    # Compute expectation values in parallel
    expectation_values = Parallel(n_jobs=-1)(
        delayed(compute_expectation)(pauli_op) for pauli_op in sparse_pauli_ops
    )

    # Create a dictionary to map Pauli operators to their expectation values
    pauli_expectation_dict = {
        pauli_op: value for pauli_op, value in zip(unique_pauli_ops, expectation_values)
    }
    print(pauli_expectation_dict)
    
    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Post-processing: Compute RDM elements using the precomputed expectation values
    for idx in unique_indices:
        i, j, k, l = idx

        # Skip invalid indices
        if is_invalid_index(i, j, k, l, num_qubits):
            continue

        # Initialize total expectation value for this RDM element
        total_expectation_value = 0.0

        # Retrieve the Pauli terms and coefficients for this RDM element
        pauli_terms = rdm_pauli_terms.get(idx, [])

        # Sum over the Pauli terms
        for pauli_op, coeff in pauli_terms:
            expectation_value = coeff * pauli_expectation_dict[pauli_op]
            total_expectation_value += expectation_value

        # Store the calculated value in the 2-RDM matrix
        temp = np.real(total_expectation_value)
        
        # Output the calculated 2-RDM element for debugging purposes
        # print(f"Calculated 2-RDM element with Parallel after GPU Parallel 1st way, ({i}, {j}, {k}, {l}) is {temp}")
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm



def get_one_rdm_term_wavefunction(i, j, num_qubits, ground_state_wavefunction):
    """
    Calculate a specific element of the one-body reduced density matrix (1-RDM) 
    using the ground state wavefunction.

    Parameters:
    -----------
    i, j : int
        Indices for the 1-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (i, j, float)
        A tuple containing the indices (i, j) and the calculated value of the 1-RDM element.
    """
    # Check if the indices are out of bounds for the system size
    if i >= num_qubits or j >= num_qubits:
        # Return 0 if indices are invalid
        return i, j, 0.0
    
    # Create the fermionic operator for the 1-RDM element (creation at i, annihilation at j)
    fermi_term = FermionicOp({f"+_{i} -_{j}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Compute the expectation value using the ground state wavefunction
    # If ground_state_wavefunction is a density matrix, we use the trace
    # Otherwise, it's assumed to be a state vector, and we compute ⟨ψ|O|ψ⟩
    qubit_matrix = qubit_term.to_matrix()
    
    if ground_state_wavefunction.ndim == 2:
        # Assume it's a density matrix (mixed state), calculate trace(ρO)
        temp = np.trace(ground_state_wavefunction @ qubit_matrix)
    else:
        # Assume it's a pure state, calculate ⟨ψ|O|ψ⟩
        temp = ground_state_wavefunction.conj().T @ qubit_matrix @ ground_state_wavefunction
    
    # Take the real part of the result
    temp = np.real(temp)
    
    # print(f"Calculated 1-RDM element with wavefunction ({i}, {j}) is {temp}")
    
    return i, j, temp

@measure_execution_time
def get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction):
    """
    Calculate the full one-body reduced density matrix (1-RDM) using the ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    one_rdm : np.ndarray
        A 2D numpy array representing the one-body reduced density matrix.
    """
    # Ensure the ground state wavefunction is provided
    if ground_state_wavefunction is None:
        raise ValueError("Ground state wavefunction must be provided.")
    
    # Initialize an empty matrix for the 1-RDM
    one_rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    
    # Use functools.partial to fix constant parameters for the 1-RDM element calculation
    func = functools.partial(get_one_rdm_term_wavefunction, num_qubits=num_qubits, ground_state_wavefunction=ground_state_wavefunction)
    
    # Generate all possible index combinations (i, j) for the 1-RDM
    indices = [(i, j) for i in range(num_qubits) for j in range(num_qubits)]
    
    # Parallel computation of all 1-RDM elements (set n_jobs according to your system)
    results = Parallel(n_jobs=-1)(delayed(func)(i, j) for i, j in indices)

    # Fill the 1-RDM matrix with the computed results
    for result in results:
        i, j, temp = result
        one_rdm[i, j] = temp
        one_rdm[j, i] = temp
    return one_rdm

def get_two_rdm_term_wavefunction(i, j, k, l, num_qubits, ground_state_wavefunction):
    """
    Calculate a specific element of the two-body reduced density matrix (2-RDM)
    using the ground state wavefunction.

    Parameters:
    -----------
    i, j, k, l : int
        Indices for the 2-RDM element.
    
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (i, j, k, l, float)
        A tuple containing the indices (i, j, k, l) and the calculated value of the 2-RDM element.
    """
    # Check if the indices are invalid based on system size or symmetry
    if is_invalid_index(i, j, k, l, num_qubits):
        return i, j, k, l, 0.0
    
    # Create the fermionic operator for the 2-RDM element (two creation, two annihilation operators)
    fermi_term = FermionicOp({f"+_{i} +_{j} -_{k} -_{l}": 1}, num_spin_orbitals=num_qubits)
    
    # Map the fermionic operator to a qubit operator using the Jordan-Wigner transformation
    mapper = JordanWignerMapper()
    qubit_term = mapper.map(fermi_term)
    
    # Convert the qubit operator to a matrix
    qubit_matrix = qubit_term.to_matrix()
    
    # Compute the expectation value using the ground state wavefunction
    if ground_state_wavefunction.ndim == 2:
        # If it's a density matrix, calculate Tr(ρO)
        temp = np.trace(ground_state_wavefunction @ qubit_matrix)
    else:
        # If it's a pure state, calculate ⟨ψ|O|ψ⟩
        temp = ground_state_wavefunction.conj().T @ qubit_matrix @ ground_state_wavefunction

    # Take the real part of the result
    temp = np.real(temp)
    
    # print(f"Calculated 2-RDM element with wavefunction ({i}, {j}, {k}, {l}) is {temp}")
    
    return i, j, k, l, temp

@measure_execution_time
def get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction):
    """
    Calculate the full two-body reduced density matrix (2-RDM) using the ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of spin orbitals (qubits) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    two_rdm : np.ndarray
        A 4D numpy array representing the two-body reduced density matrix.
    """
    # Ensure that the ground state wavefunction is provided
    if ground_state_wavefunction is None:
        raise ValueError("Ground state wavefunction must be provided.")
    
    # Get the unique index combinations for the 2-RDM (reducing unnecessary calculations)
    unique_indices = get_two_rdm_reduced_indices(num_qubits)
    
    # Use functools.partial to fix the constant parameters (num_qubits, ground_state_wavefunction)
    func = functools.partial(get_two_rdm_term_wavefunction, num_qubits=num_qubits, ground_state_wavefunction=ground_state_wavefunction)

    # Use parallel computing to compute 2-RDM elements
    results = Parallel(n_jobs=-1)(delayed(func)(i, j, k, l) for i, j, k, l in unique_indices)

    # Initialize an empty 4D matrix to store the 2-RDM elements
    two_rdm = np.zeros((num_qubits, num_qubits, num_qubits, num_qubits), dtype=complex)

    # Fill the 2-RDM matrix with the computed results, including symmetries
    for result in results:
        i, j, k, l, temp = result
        
        # Assign calculated value to the corresponding positions in the 2-RDM matrix
        two_rdm[i, j, k, l] = temp
        two_rdm[k, l, i, j] = temp       # Symmetry: 2-RDM[i,j,k,l] = 2-RDM[k,l,i,j]
        two_rdm[j, i, k, l] = -temp      # Antisymmetry in i, j indices
        two_rdm[i, j, l, k] = -temp      # Antisymmetry in k, l indices

    return two_rdm

def create_gate_circuit(gate_name: str) -> QuantumCircuit:
    """
    创建一个单量子比特线路，并应用指定的门。
    """
    circuit = QuantumCircuit(1)
    gate_map = {
        'h': circuit.h, 'x': circuit.x, 'y': circuit.y, 
        'z': circuit.z, 's': circuit.s, 'sdg': circuit.sdg
    }
    if gate_name in gate_map:
        gate_map[gate_name](0)
    else:
        raise ValueError(f"Gate '{gate_name}' is not supported by this helper.")
    return circuit

def find_closest_gate(original_gate_op: Operator) -> object:
    """
    通过比较过程保真度，找到与给定原始门最接近的Clifford门。
    """
    # 候选的Clifford门集合 (增加了SdgGate)
    clifford_gate_classes = {'h': HGate, 'x': XGate, 'y': YGate, 'z': ZGate, 's': SGate, 'sdg': SdgGate}
    
    max_fidelity = -1
    closest_gate_name = None

    # 遍历所有候选Clifford门，计算保真度
    for gate_name, gate_class in clifford_gate_classes.items():
        clifford_circuit = create_gate_circuit(gate_name)
        fidelity = process_fidelity(original_gate_op, Operator(clifford_circuit))
        
        if fidelity > max_fidelity:
            max_fidelity = fidelity
            closest_gate_name = gate_name
            
    # 返回保真度最高的门的实例
    return clifford_gate_classes[closest_gate_name]()


def replace_gates_optimal(original_circuit, optimal_parameters_noiseless):
    """
    遍历线路，用保真度最高的Clifford门替换非Clifford的单比特门，并进行验证。
    """
    # 定义我们的目标Clifford门集合
    clifford_gate_names = {'h', 'x', 'y', 'z', 's', 'sdg', 'i', 'cx'} # cx是多比特门，也算Clifford
    single_qubit_gate_names_for_count = {'h', 's', 'sdg', 'z', 'i', 'rz', 'x', 'y', 't', 'tdg', 'u', 'p'}

    gate_set = {instruction.operation.name for instruction in original_circuit.data}
    if not gate_set.issubset({'cx', 'h', 's', 'rz'}):
        # Transpile the circuit to use only the allowed gates
        original_circuit = transpile(original_circuit, basis_gates=['cx', 'h', 's', 'rz'])
        
    # --- 验证前计数 ---
    original_single_gates_count = sum(
        1 for instruction in original_circuit.data 
        if instruction.operation.name in single_qubit_gate_names_for_count
    )
    print(f"Number of single-qubit gates before replacement: {original_single_gates_count}")

    # Count the number of each type of gate in the original circuit
    gate_counts = {gate: sum(1 for instruction in original_circuit.data if instruction.operation.name == gate)
                   for gate in ['cx', 'h', 's', 'x', 'y', 'z',  'sdg', 'i', 'rz']}
    
    # Print gate counts for the original circuit
    for gate, count in gate_counts.items():
        print(f"Number of '{gate}' gates before replacing gates: {count}")
        
    # 创建新线路
    new_circuit = QuantumCircuit(original_circuit.num_qubits, name="Fidelity_Optimized_" + original_circuit.name)
    replaced_gates_num = 0

    for instruction in original_circuit.data:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        
        # --- 核心替换逻辑 ---
        # 只处理单比特门，且该门不是我们定义的Clifford门之一
        is_single_qubit_gate = len(qargs) == 1
        is_non_clifford = instr.name not in clifford_gate_names

        if is_single_qubit_gate and is_non_clifford:
            # 1. 将当前门转换为Operator对象
            temp_gate_qc = QuantumCircuit(1)
            temp_gate_qc.append(instr, [0])
            original_gate_op = Operator(temp_gate_qc)

            # 2. 使用您的函数找到最近的Clifford门
            closest_gate = find_closest_gate(original_gate_op)
            
            # 3. 在新线路中添加替换后的门
            new_circuit.append(closest_gate, qargs, cargs)
            replaced_gates_num += 1
        else:
            # 如果是多比特门或已经是Clifford门，直接保留
            new_circuit.append(instr, qargs, cargs)

    print(f"Total non-Clifford single-qubit gates replaced: {replaced_gates_num}")
    
    # --- 验证后计数 ---
    new_single_gates_count = sum(
        1 for instruction in new_circuit.data 
        if instruction.operation.name in single_qubit_gate_names_for_count
    )
    print(f"Number of single-qubit gates after replacement: {new_single_gates_count}")
    
    # Count the number of each type of gate in the original circuit
    gate_counts = {gate: sum(1 for instruction in new_circuit.data if instruction.operation.name == gate)
                   for gate in ['cx', 'h', 's',  'x', 'y', 'z',  'sdg', 'i','rz']}
    
    # Print gate counts for the original circuit
    for gate, count in gate_counts.items():
        print(f"Number of '{gate}' gates after replacing gates: {count}")
        
    # --- 执行检查 ---
    if original_single_gates_count == new_single_gates_count:
        print("Verification successful: Single-qubit gate count is consistent.")
        return new_circuit, replaced_gates_num
    else:
        print("Error: Verification failed. Single-qubit gate count mismatch.")
        return None, 0

def check_clifford_and_get_stabilizer(circuit: QuantumCircuit) -> str:
    """
    Check if the given quantum circuit is a Clifford circuit.
    If it is a Clifford circuit, return its stabilizers.
    If it is not a Clifford circuit, return a message indicating so.

    Parameters:
    -----------
    circuit : QuantumCircuit
        The quantum circuit to check for Clifford property.
        
    Returns:
    --------
    str
        A message indicating whether the circuit is a Clifford circuit or not,
        and if it is, returns the stabilizer group of the circuit.
    """
    try:
        # Convert the quantum circuit to a Clifford object
        cliff = Clifford(circuit)
        
        # Get the stabilizers of the Clifford circuit
        stabilizer = cliff.to_labels(mode="S")  # "S" mode returns stabilizers in label form
        
        # Return a message with the stabilizers
        return f"This is a Clifford circuit. The stabilizers are: {stabilizer}"
    
    except QiskitError as e:
        # Catch the error if the circuit is not a Clifford circuit
        return f"This is not a Clifford circuit. Error: {str(e)}"

    
def RDM_with_given_wavefunction(num_qubits: int, ground_state_wavefunction: np.ndarray):
    """
    Calculate both the one-body and two-body reduced density matrices (RDMs)
    using a given ground state wavefunction.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    ground_state_wavefunction : np.ndarray
        The ground state wavefunction represented as a density matrix or state vector.

    Returns:
    --------
    tuple : (np.ndarray, np.ndarray)
        The one-body RDM and two-body RDM calculated from the given wavefunction.
    """
    # Compute the one-body reduced density matrix using the ground state wavefunction
    one_RDM_with_given_wavefunction = get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction)
    
    # Compute the two-body reduced density matrix using the ground state wavefunction
    two_RDM_with_given_wavefunction = get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction)
    
    return one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction

def RDM_with_given_circuit(num_qubits: int, circuit: QuantumCircuit, estimator: Estimator):
    """
    Calculate both the one-body and two-body reduced density matrices (RDMs)
    using a given quantum circuit and estimator.

    Parameters:
    -----------
    num_qubits : int
        The number of qubits (spin orbitals) in the system.
    
    circuit : QuantumCircuit
        The quantum circuit used to prepare the quantum state for measurement.
    
    estimator : Estimator
        The estimator object (either noiseless or noisy) used to compute expectation values.

    Returns:
    --------
    tuple : (np.ndarray, np.ndarray)
        The one-body RDM and two-body RDM calculated from the given quantum circuit.
    """
    # Compute the two-body reduced density matrix using the quantum circuit and estimator
    two_rdm_given_circuit = get_two_rdm(num_qubits, circ=circuit, estimator=estimator)
    
    # Compute the one-body reduced density matrix using the quantum circuit and estimator
    one_rdm_given_circuit = get_one_rdm(num_qubits, circ=circuit, estimator=estimator)
    
    return one_rdm_given_circuit, two_rdm_given_circuit

def VQE_RDM_thm_noisy_RG(molecule: str, d:str, num_qubits: int, n_particles: int, prob_1: float, prob_2: float, shots_num: int, HEAp: int):
    """
    执行无噪声和有噪声的VQE，并返回计算得到的1-RDM和2-RDM。

    参数:
    molecule : str
        分子名称
    d : float
        分子距离
    num_qubits : int
        量子比特数
    n_particles : int
        粒子数
    prob_1 : float
        1-比特门的噪声概率
    prob_2 : float
        2-比特门的噪声概率
    shots_num : int
        模拟中的shots数量

    返回:
    tuple: (1-RDM和2-RDM的多组结果)
    """
    _, noisy_estimator = build_noisy_estimator(prob_1=prob_1, prob_2=prob_2, shots_num=shots_num)

    # 文件名
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    try:
        with open(full_input_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"


    qubit_hamiltonian = data['qubit_hamiltonian']
    n_particles = data['n_particles']

    # 生成无噪声和有噪声的 ansatz 和 VQE 结果
    hamiltonian_qiskit = Gene_Qiskit_VQE_hamiltonian(num_qubits, qubit_hamiltonian)
    
    result_noiseless, _, circ_noiseless, estimator_noiseless, optimal_parameters_noiseless = HEA_VQE_noiseless(
        hamiltonian_qiskit=hamiltonian_qiskit, num_qubits=num_qubits, n_particles=n_particles, HEAp=HEAp, max_iterations=5000)
    print(result_noiseless)
    
    result_noisy, _, circ_noisy, _, _ = HEA_VQE_noisy(
        hamiltonian_qiskit=hamiltonian_qiskit, num_qubits=num_qubits, n_particles=n_particles, noisy_estimator=noisy_estimator, HEAp=HEAp, max_iterations=5000)
    print(result_noisy)
    
    # two_rdm_noisy_Parallel = get_two_rdm_Parallel(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    # two_rdm_noisy_Parallel2 = get_two_rdm_Parallel2(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    # two_rdm_noisy_Parallel3 = get_two_rdm_Parallel3(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    
    # 生成并优化 ansatz
    circ_rgo, replaced_gates_num = replace_gates_optimal(circ_noiseless, optimal_parameters_noiseless)
    # 基态波函数生成并计算RDM
    RG_ground_state_wavefunction = Statevector(circ_rgo).data
    RG_ground_state_wavefunction = to_density_matrix(RG_ground_state_wavefunction)
    
    one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction = RDM_with_given_wavefunction(num_qubits, RG_ground_state_wavefunction)
    
    one_rdm_given_circuit, two_rdm_given_circuit = RDM_with_given_circuit(num_qubits, circ_rgo, noisy_estimator)


    # 使用无噪声和有噪声量子电路的 RDM 结果
    ground_state_wavefunction_noiseless = Statevector(circ_noiseless).data
    ground_state_wavefunction_noiseless = to_density_matrix(ground_state_wavefunction_noiseless)
    
    two_rdm_noiseless = get_two_rdm(num_qubits, circ=circ_noiseless, estimator = estimator_noiseless)

    two_rdm_thm = get_two_rdm_wavefunction(num_qubits, ground_state_wavefunction=ground_state_wavefunction_noiseless)
    two_rdm_noisy = get_two_rdm(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    # two_rdm_noisy_Parallel = get_two_rdm_Parallel(num_qubits, circ=circ_noisy, estimator=noisy_estimator)
    one_rdm_noiseless = get_one_rdm(num_qubits, circ=circ_noiseless, estimator = estimator_noiseless)
    one_rdm_thm = get_one_rdm_wavefunction(num_qubits, ground_state_wavefunction=ground_state_wavefunction_noiseless)
    one_rdm_noisy = get_one_rdm(num_qubits, circ=circ_noisy, estimator=noisy_estimator)

    return (one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction, 
            one_rdm_given_circuit, two_rdm_given_circuit,
            one_rdm_noiseless, two_rdm_noiseless,
            one_rdm_noisy, two_rdm_noisy,
            one_rdm_thm,two_rdm_thm,result_noiseless,result_noisy, replaced_gates_num)

def calculate_and_display_rdm_table(rdm_list, labels):
    """
    计算并生成 RDM 的两两 Frobenius 范数对比表格，横纵坐标为 RDM 简写标签。

    参数:
    rdm_list : list
        包含所有要比较的 RDM 矩阵。
    labels : list
        每个 RDM 的简写名称，用于表格显示。

    输出:
    Pandas DataFrame 显示并保存为 CSV 文件。
    """
    n = len(labels)
    
    # 创建空矩阵存储范数
    norm_matrix = np.zeros((n, n))

    # 两两组合计算 Frobenius 范数，并填充矩阵
    for (i, j) in combinations(range(n), 2):
        norm, _ = calculate_frobenius_norm_difference(rdm_list[i], rdm_list[j])
        norm_matrix[i, j] = norm
        norm_matrix[j, i] = norm  # 对称填充

    # 创建 Pandas DataFrame 生成表格
    df = pd.DataFrame(norm_matrix, index=labels, columns=labels)
    
    return df
def process_distance(d, molecule, prob_1, prob_2, shots_num, HEAp):
    """
    处理单个距离d值的所有计算和数据保存任务。
    这是一个独立的“工作单元”，非常适合并行化。
    """
    print(f"--- [START] Processing for d = {d:.1f} ---")
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    
    # 3. 在 open() 中使用构建好的完整路径
    print(f"Attempting to load file: {full_input_path}")
    try:
        with open(full_input_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"

    try:
        with open(input_filename, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found for d = {d:.2f}. Skipping this distance.")
        print(f"Missing file: {input_filename}")
        return f"Failed for d={d:.2f}"

    # 访问变量
    molecular_hamiltonian = data['molecular_hamiltonian']
    qubit_hamiltonian = data['qubit_hamiltonian']
    FCI_val = data['FCI_val']
    one_body_coefficients = data['one_body_coefficients']
    two_body_coefficients = data['two_body_coefficients']
    constant = data['constant']
    n_elec = data['n_elec']
    n_particles = data['n_particles']
    num_qubits = data['num_qubits']

    # 2. 执行核心计算
    
    (one_RDM_with_given_wavefunction, two_RDM_with_given_wavefunction, 
     one_rdm_given_circuit, two_rdm_given_circuit,
     one_rdm_noiseless, two_rdm_noiseless,
     one_rdm_noisy, two_rdm_noisy,
     one_rdm_thm, two_rdm_thm,result_noiseless,result_noisy, replaced_gates_num) = VQE_RDM_thm_noisy_RG(
        molecule=molecule, d=d, num_qubits=num_qubits, n_particles=n_particles, 
        prob_1=prob_1, prob_2=prob_2, shots_num=shots_num, HEAp = HEAp
    )
    
    # 3. 后处理和数据整理
    one_rdm_list = [one_RDM_with_given_wavefunction, one_rdm_given_circuit, one_rdm_noiseless, one_rdm_noisy, one_rdm_thm]
    two_rdm_list = [two_RDM_with_given_wavefunction, two_rdm_given_circuit, two_rdm_noiseless, two_rdm_noisy, two_rdm_thm]
    labels = ['WGF', 'CIRC', 'NSL', 'NOI', 'THM']
    
    df_1_rdm = calculate_and_display_rdm_table(one_rdm_list, labels=labels)
    df_2_rdm = calculate_and_display_rdm_table(two_rdm_list, labels=labels)

    # 4. 使用包含d的动态文件名保存输出
    # 格式化d为两位小数，确保文件名统一美观
    d_str = f"{d:.1f}" 

    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_one_rdm_list.pkl'), 'wb') as f:
        pickle.dump(one_rdm_list, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_two_rdm_list.pkl'), 'wb') as f:
        pickle.dump(two_rdm_list, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_df_1_rdm.pkl'), 'wb') as f:
        pickle.dump(df_1_rdm, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_df_2_rdm.pkl'), 'wb') as f:
        pickle.dump(df_2_rdm, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_result_noiseless.pkl'), 'wb') as f:
        pickle.dump(result_noiseless, f)
    with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_result_noisy.pkl'), 'wb') as f:
        pickle.dump(result_noisy, f)
    print(f"--- [DONE]  Processing for d = {d:.2f}. Data saved. ---")

    return replaced_gates_num

# ==============================================================================

# Part B


'''
def is_invalid_index(i: int, j: int, k: int, l: int, num_spin_orbitals: int) -> bool:
    """
    Check if the given indices are invalid based on the number of spin orbitals.

    Args:
        i: The first index.
        j: The second index.
        k: The third index.
        l: The fourth index.
        num_spin_orbitals: The total number of spin orbitals.

    Returns:
        True if the indices are invalid, False otherwise.
    """
    valid_term = (
        (i < num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i < num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l >= num_spin_orbitals // 2) or
        (i < num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k < num_spin_orbitals // 2 and l >= num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j < num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l < num_spin_orbitals // 2) or
        (i >= num_spin_orbitals // 2 and j >= num_spin_orbitals // 2 and k >= num_spin_orbitals // 2 and l >= num_spin_orbitals // 2)
    )
    return not valid_term
'''
def check_two_rdm_standard(two_rdm, num_spin_orbitals):
    for i in range(num_spin_orbitals):
        for j in range(num_spin_orbitals):
            for k in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    if is_invalid_index(i, j, k, l, num_spin_orbitals) and two_rdm[i, j, k, l] != 0:
                        return False
    return True

def kron_to_ikjl(two_rdm_ref_ij_kl):
    """
    将一个经过压缩或重塑为二维数组的四阶张量转换回其原始的四维形式，
    并将其元素从"ij-kl"格式排列改为"ik-jl"格式。

    输入:
    - two_rdm_ref_ij_kl (numpy.ndarray): 一个二维数组，表示一个被压缩或重塑的四阶张量，
                                          形状为(num_spin_orbitals^2, num_spin_orbitals^2)，
                                          其中 num_spin_orbitals 是自旋轨道的数量。

    输出:
    - numpy.ndarray: 一个四维张量，其元素根据"ik-jl"格式排列，
                     形状为(num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals)。
    """

    # 计算自旋轨道的数量
    num_spin_orbitals = int(np.sqrt(two_rdm_ref_ij_kl.shape[0]))

    # 初始化一个新的四维张量来存储重排后的数据
    two_rdm_ref_ikjl = np.ndarray((num_spin_orbitals,) * 4)

    # 遍历每个索引，重新排列张量的元素
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 重新排列为"ik-jl"格式
                    two_rdm_ref_ikjl[i, k, j, l] = two_rdm_ref_ij_kl[i*num_spin_orbitals + j, k*num_spin_orbitals + l].value

    return two_rdm_ref_ikjl

def ikjl_to_kron(two_rdm_ref_ikjl):
    """
    将一个按照"ik-jl"格式排列的四阶张量转换为按照"ij-kl"格式排列的二维数组。

    输入:
    - two_rdm_ref_ikjl (numpy.ndarray): 一个四维张量，按照"ik-jl"格式排列，
                                         形状为(num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals)，
                                         其中 num_spin_orbitals 是自旋轨道的数量。

    输出:
    - numpy.ndarray: 一个二维数组，按照"ij-kl"格式排列，
                     形状为(num_spin_orbitals^2, num_spin_orbitals^2)。
    """

    # 获取自旋轨道的数量
    num_spin_orbitals = two_rdm_ref_ikjl.shape[0]

    # 初始化一个新的二维数组来存储重排后的数据
    two_rdm_ref_ij_kl = np.ndarray((num_spin_orbitals**2,)*2)

    # 遍历每个索引，重新排列张量的元素
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 重新排列为"ij-kl"格式
                    two_rdm_ref_ij_kl[i*num_spin_orbitals + j, k*num_spin_orbitals + l] = two_rdm_ref_ikjl[i, k, j, l]

    return two_rdm_ref_ij_kl

def kron_to_ijkl(two_rdm_ref_ij_kl):
    return np.einsum('ikjl->ijkl',kron_to_ikjl(two_rdm_ref_ij_kl))

def expr_to_val (M_ikjl):
    return np.apply_along_axis(lambda arr: np.array([expr.value for expr in arr]), M_ikjl.ndim-1,M_ikjl)

def validate_inputs(two_rdm, num_particles, num_spin_orbitals):
    """
    验证输入的有效性，包括two_rdm的维度和旋轨道数是否符合条件。
    """
    assert two_rdm.ndim == 4, 'two_rdm given is not a fourth-order tensor'
    assert num_spin_orbitals >= num_particles, 'the number of available orbitals should be no less than the number of fermions'

def prepare_hamiltonian(one_body_coefficients, two_body_coefficients, two_rdm):
    """
    准备哈密顿量的一体和二体系数。
    """
    h1_ij = np.array(one_body_coefficients) if one_body_coefficients is not None else None
    h2_ijkl = np.array(two_body_coefficients) if two_body_coefficients is not None else None
    if h2_ijkl is not None:
        assert h2_ijkl.shape == two_rdm.shape, 'the shape of h2_ijkl should be the same as two_rdm'
    return h1_ij, h2_ijkl

def check_tensor_convention(two_rdm):
    """
    检查并调整two_rdm的索引约定。
    """
    num_particles_ref = np.einsum('ijij', two_rdm)
    flag_ijkl = num_particles_ref > 0
    num_particles_ref = (1 + np.sqrt(1 + 4 * abs(num_particles_ref))) / 2
    print(f"Initially {num_particles_ref} particles measured")

    if not flag_ijkl:
        two_rdm = np.einsum('ijlk->ijkl', two_rdm)
        print("Index convention: ijlk. Changing the input tensor convention from ijlk->ijkl.")
        print("The given two-rdm tensor is of index ijlk, i.e.a two electron wavefunction 1,2 would be encoded as 1,2,2,1")
        print("This function would not change the convention between input and output tensor")
    else:
        print("Index convention: ijkl. Not changing the input tensor convention.")
        print("The given two-rdm tensor is of index ijkl, i.e. a two electron wavefunction 1,2 would be encoded as 1,2,1,2")
        print("This function would not change the convention between input and output tensor")         
    return two_rdm, flag_ijkl
def to_kron_with_constraint(M_ikjl, constraints, pos_def=True):
    ##return the two dimensional alias of the matrix with equality and positive definiteness constraint
    offset= M_ikjl.shape[0]
    ## group the (i,j) as a composite index and (k,l) as a composite index
    M_ij_kl = cp.Variable((offset**2, offset**2))
    assert M_ij_kl.is_matrix(), 'should convert to a two dimensional cvxpy matrix'
    
    for i in range (offset):
        for k in range(offset):
            for j in range(offset):
                for l in range (offset):
                      constraints.append(M_ij_kl[i*offset+j][k*offset+l] == M_ikjl[i,k][j,l])
    if pos_def:
        constraints.append(M_ij_kl>>0)
        
    return M_ij_kl, constraints


def symmetrize_2rdm (M, constraints):   
    if type(M)==np.ndarray:
        offset= M.shape[0]
    if type(M)==cp.Variable:
        offset= int(np.sqrt(M.shape[0]))  
    for i in range (offset):
        for k in range(offset):
            for j in range(offset):
                for l in range (offset):
                    if type(M)==np.ndarray:
                        ## M is forth order tensor in ikjl index
                            if i<j : constraints.append(M[i,k][j,l]== - M[j,k][i,l])
                            if k<l : constraints.append(M[i,k][j,l]== - M[i,l][j,k])
                            if (i,j)<(k,l): constraints.append(M[i,k][j,l]== M[k,i][l,j])
                    if type(M)==cp.Variable:
                        ## M is a kronecker product two-dimensional matrix in (i*offset+j, k*offset+l) index
                            if i<j :constraints.append(M[i*offset+j][k*offset+l]== - M[j*offset+i][k*offset+l])
                            if k<l :constraints.append(M[i*offset+j][k*offset+l]== - M[i*offset+j][l*offset+k])
                            if (i,j)<(k,l): constraints.append(M[i*offset+j][k*offset+l]== M[k*offset+l][i*offset+j])                       
    return M,constraints
def check_2rdm_properties(tensor):
    offset = tensor.shape[0]
    for i in range(offset):
        for j in range(offset):
            for k in range(offset):
                for l in range(offset):
                    # Check Hermitian property
                    if tensor[i, j, k, l] != tensor[k, l, i, j]:
                        print(f"不满足厄米性质", (i, j, k, l), (k, l, i, j))

                    # Check antisymmetry property
                    if i != j and tensor[i, j, k, l] != -tensor[j, i, k, l]:
                        print(f"不满足反对称性质", {(i, j, k, l)}, {(j, i, k, l)})
                    if k != l and tensor[i, j, k, l] != -tensor[i, j, l, k]:
                        print(f"不满足反对称性质", {(i, j, k, l)}, {(i, j, l, k)})
    print(f"=================================================================")
    return "满足"

def optimize(two_rdm, num_particles, one_body_coefficients=None, two_body_coefficients=None, epsilon=None, max_iters=20000, verbose=True, accuracy=1e-10):
    
    # 检查张量顺序和Pauli排斥原理
    two_rdm = np.array(two_rdm)
    num_spin_orbitals = two_rdm.shape[0]
    constraints = []

    # 验证输入
    validate_inputs(two_rdm, num_particles, num_spin_orbitals)

    # 检查是否需要最小化能量
    flag_to_minimize_E = bool(two_body_coefficients is not None and one_body_coefficients is not None)

    # 准备哈密顿量
    h1_ij, h2_ijkl = prepare_hamiltonian(one_body_coefficients, two_body_coefficients, two_rdm)
    
    # 检查张量约定
    two_rdm, flag_ijkl = check_tensor_convention(two_rdm)

    # 转换张量
    
    is_standard = check_two_rdm_standard(two_rdm, num_spin_orbitals)
    
    print(f"two_rdm is {is_standard}")
    
    two_rdm_ref_ikjl = np.einsum('ijkl->ikjl', two_rdm)
    
    
    print(np.shape(two_rdm_ref_ikjl))
    
    two_rdm_ref_ij_kl = cp.Constant(ikjl_to_kron(two_rdm_ref_ikjl))

    # 生成 Kronecker Delta 张量
    
    kd_ijkl = np.einsum('ij,kl->ijkl', np.eye(num_spin_orbitals), np.eye(num_spin_orbitals))
    
    two_rdm_ref_ikjl_echo = np.einsum('ijkl->ikjl',two_rdm)
    
    two_rdm_ref_ij_kl_echo = ikjl_to_kron(two_rdm_ref_ikjl_echo)

    ##prepare in cvxpy constant form
    ##kronecker delta ijkl
    
    kd_ijkl= np.einsum('ij,kl->ijkl',np.eye(num_spin_orbitals),np.eye(num_spin_orbitals))
    
    ##define an 2d nparray of 2d cvxpy variable object in the index of ik jl respectively
    ##since cvxpy does not support variable with dimensions greater than 2
    
    var_ls= []
    for i in range (num_spin_orbitals):
        temp=[]
        for k in range (num_spin_orbitals):
            temp.append(cp.Variable((num_spin_orbitals, num_spin_orbitals)))
        var_ls.append(temp)
        
    two_rdm_new_ikjl= np.array(var_ls, dtype=object)
    
    print(np.shape(two_rdm_new_ikjl))
    
    for i in range(num_spin_orbitals):
        for k in range(num_spin_orbitals):
            for j in range(num_spin_orbitals):
                for l in range(num_spin_orbitals):
                    # 如果参考矩阵在(i, k, j, l)位置的值为0，则在优化变量矩阵中设置相应的约束
                    if i == j or l == k or is_invalid_index(i, j, k, l, num_spin_orbitals):
                        # print(two_rdm_ref_ikjl[i][k][j][l])
                        # print(two_rdm_new_ikjl[i][k][j][l])
                        constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                    # if i == j or l == k or invalid(i, j, k, l, num_spin_orbitals):
                    #    constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                        
    ##tracing to give one_rdm_new # Test_one_RDM = np.einsum('prrq', FCI_two_RDM)np.isclose(two_rdm_ref_ikjl[i][k][j][l], 0, atol=1e-8):
                       # constraints.append(two_rdm_new_ikjl[i][k][j][l] == cp.Constant(0))
                    # 您之前的条件可以保留，或者根据需要进行修改
                    # elif
    
    one_rdm_new=  1/(num_particles-1)*(np.vectorize(cp.trace)(two_rdm_new_ikjl))                        
    one_rdm_ref=  1/(num_particles-1)*np.trace(two_rdm_ref_ikjl)
    
    ## compute the Q-matrix in Q_ikjl 
    ## Q_ikjl = 2rdm_ikjl + kd_ijkl -kd_ilkj - 1rdm_ij kd_kl -1rdm_kl kd_ij + 1rdm_jk kd_il + 1rdm_il kd_jk
    
    kd_ij_1rdm_kl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    
    for i in range (num_spin_orbitals):
        for j in range (num_spin_orbitals):
            for k in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                    if i==j:
                        kd_ij_1rdm_kl[i,j,k,l]= one_rdm_new[k,l]
                    else:
                        kd_ij_1rdm_kl[i,j,k,l]= cp.Constant(0)
                        
    Q_ikjl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    for i in range (num_spin_orbitals):
        for k in range (num_spin_orbitals):
            for j in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                        Q_ikjl[i,k,j,l]= two_rdm_new_ikjl[i,k][j,l] + kd_ijkl[j,l,i,k]-kd_ijkl[i,l,j,k]\
                            - kd_ij_1rdm_kl[j,l,i,k]-kd_ij_1rdm_kl[i,k,j,l]+kd_ij_1rdm_kl[i,l,j,k]+kd_ij_1rdm_kl[j,k,i,l]
                            
    ##compute the G-matrix in G_ikjl
    ##G_ikjl = kd_kl_1rdm_ij - 2rdm_iljk
    
    G_ikjl = np.ndarray((num_spin_orbitals,)*4,dtype=object)
    for i in range (num_spin_orbitals):
        for k in range (num_spin_orbitals):
            for j in range (num_spin_orbitals):
                for l in range (num_spin_orbitals):
                        G_ikjl[i,k,j,l]= kd_ij_1rdm_kl[j,l,i,k]-two_rdm_new_ikjl[i,k][l,j]

    ##symmetry constraints
    ##symmetrizing 2rdm should automatically enforce symmetry of 1rdm, P, and Q
    
    two_rdm_new_ikjl, constraints = symmetrize_2rdm(two_rdm_new_ikjl, constraints)

    ##Positivity constraints
                
    two_rdm_new_ij_kl, constraints = to_kron_with_constraint(two_rdm_new_ikjl, constraints)
   
    G_ij_kl, constraints = to_kron_with_constraint(G_ikjl, constraints)
    Q_ij_kl, constraints = to_kron_with_constraint(Q_ikjl, constraints)
    
    ##Trace constraints
    
    ##Tr{2rdm}=N*(N-1) this should automatically enforce the trace of 1rdm, P, and Q
    
    constraints.append(cp.trace(two_rdm_new_ij_kl)==num_particles*(num_particles-1))
    
    ##set up the objective
    ##to minimize the Frobenius norm between the optimized 2rdm and the reference 2rdm 
    
    if flag_to_minimize_E:
        if epsilon is not None:
            if np.isscalar(epsilon):
                # epsilon 是一个标量
                constraints.append(cp.norm(two_rdm_new_ij_kl - two_rdm_ref_ij_kl, 'fro') <= epsilon)                
            elif epsilon.shape == two_rdm_new_ij_kl.shape:                
                n_dim = two_rdm_new_ij_kl.shape[0]
                for i in range(n_dim):
                    for j in range(n_dim):
                        if epsilon[i][j] != 0 and np.any(two_rdm_ref_ij_kl_echo[i][j] != 0):
                            epsilon_ij_kl = cp.Constant(np.abs(epsilon[i][j]))
                            two_rdm_ref_ij_kl_echo_new = cp.Constant(two_rdm_ref_ij_kl_echo[i][j])                                     
                            constraints.append(cp.norm(two_rdm_new_ij_kl[i,j] - two_rdm_ref_ij_kl_echo_new,1) <= epsilon_ij_kl)
            else:
                # epsilon 的维度既不是标量也不与 two_rdm_new_ij_kl 一致
                raise ValueError("epsilon 的维度不正确。它必须是一个标量或与 two_rdm_new_ij_kl 的维度一致。")
                               
        h2_ikjl=np.einsum('ijkl->ikjl',h2_ijkl)
        
        h2_ij_kl=cp.Constant(ikjl_to_kron(h2_ikjl))
        
        E = np.trace(np.matmul(h1_ij,one_rdm_new))+ 0.5*cp.trace(h2_ij_kl@two_rdm_new_ij_kl)
        
        # E = np.trace(np.matmul(h1_ij,one_rdm_new))+0.5*np.trace(np.matmul(h2_ij_kl,two_rdm_new_ij_kl))
        objective = cp.Minimize(E)
        
        print("the objective is to find the lowest eigenvalue subject to n-representability and closeness to measurements")
    else:
        objective= cp.Minimize(cp.norm(two_rdm_new_ij_kl - two_rdm_ref_ij_kl, 'fro'))
        print("the objective is the find the nearest n-representable state")
    ##check if all constraints are legal
    
    assert np.all(list(map(lambda c: c.is_dcp(),constraints))), 'all constraints should be disciplined convex programming'
    print('all constraints checked to be disciplined convex programming')

    ##check if the objective is legal
    assert objective.is_dcp(), 'the objective should be disciplined convex programming'
    print('objective checked to be disciplined convex programming')

    ##create and solve the problem
    problem= cp.Problem(objective, constraints)
    
    try:
        problem.solve(verbose=verbose, solver=cp.SCS, eps=accuracy, max_iters=max_iters)
    except Exception as e:
        print(e)
        
    # 获取最优值
    optimal_value = problem.value
    
    ##summarize data
    two_rdm_res_ijkl= kron_to_ijkl(two_rdm_new_ij_kl)
    
    one_rdm_res_ij= expr_to_val(one_rdm_new)
    G_ijkl = kron_to_ijkl(G_ij_kl)
    Q_ijkl = kron_to_ijkl(Q_ij_kl)
    
    if not flag_ijkl:
        two_rdm_res_ijkl=np.einsum('ijkl->ijlk',two_rdm_res_ijkl)
        G_ijkl=np.einsum('ijkl->ijlk',G_ijkl)
        Q_ijkl=np.einsum('ijkl->ijlk',Q_ijkl)
        
        # print("please note that all forth order tensor is output in ijlk convention, not changing the convention of the input tensor.")
        
    else:
        print("please note that all forth order tensor is output in ijkl convention, not changing the convention of the input tensor.")
        
    if flag_to_minimize_E:
        if verbose:
            return E.value,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl
        else:
            return E.value,two_rdm_res_ijkl, one_rdm_res_ij
        
    if verbose:
        return two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl, optimal_value
    else:
        return two_rdm_res_ijkl, one_rdm_res_ij, optimal_value
    
def legalize_then_optimize(two_rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7, epsilon=None):
    """
    先合法化（legalize）二阶还原密度矩阵（2-RDM），然后进行优化。

    参数:
    two_rdm -- 实验得到的二阶还原密度矩阵
    num_particles -- 粒子数
    one_body_coefficients -- 单体系数
    two_body_coefficients -- 二体系数
    verbose -- 是否显示详细信息
    max_iters -- 最大迭代次数
    accuracy -- 精度
    epsilon -- 用于优化的参数

    返回:
    优化后的二阶还原密度矩阵及相关信息
    """
    two_rdm_legal, _, _, _, _ = optimize(two_rdm, num_particles, one_body_coefficients = None, two_body_coefficients = None, max_iters=max_iters, verbose=verbose, accuracy=accuracy)
    
    print("============================== Legalization was completed in legalize_then_optimize function===============================")
    
    E,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=epsilon, verbose=verbose, max_iters=max_iters, accuracy=accuracy)
    
    return E,two_rdm_res_ijkl, one_rdm_res_ij, G_ijkl, Q_ijkl

def legalized(two_rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7, epsilon=None):
    """
    合法化二阶还原密度矩阵，并返回合法化后的矩阵和其他相关信息。

    参数:
    two_rdm -- 实验得到的二阶还原密度矩阵
    num_particles -- 粒子数
    one_body_coefficients -- 单体系数
    two_body_coefficients -- 二体系数
    verbose -- 是否显示详细信息
    max_iters -- 最大迭代次数
    accuracy -- 精度

    返回:
    合法化后的二阶还原密度矩阵、epsilon 和优化后的能量值
    """
    two_rdm_legal, _, _, _, eps = optimize(two_rdm, num_particles, one_body_coefficients=None,two_body_coefficients=None, max_iters=max_iters, verbose=verbose, accuracy=accuracy)
    
    print(f"合法化后的在优化的two_rdm与two_rdm_legal的差{np.linalg.norm(two_rdm_legal - two_rdm)}，准备检查，理想值是{eps}")
    
    print("============================== Legalization was completed in legalize function===============================")
    
    E, two_rdm_res_ijkl, _, _, _ = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=eps, verbose=verbose, max_iters=max_iters, accuracy=accuracy)
         
    print(f"合法化后的在优化的two_rdm_res_ijkl与two_rdm_legal的差{np.linalg.norm(two_rdm_legal - two_rdm_res_ijkl)}，准备检查，理想值是0")
    
    print(f"合法化后的在优化的two_rdm与two_rdm_res_ijkl的差{np.linalg.norm(two_rdm_res_ijkl - two_rdm)}，准备检查，理想值是0")
    
    return two_rdm_res_ijkl, eps, E


def find_optimal_eps(molecule, prob_1, prob_2, d, shots_num, ground_state_energy, c, num_particles, accuracy, one_body_coefficients, two_body_coefficients, experimental_2rdm):

    step_size = 0.1
    threshold = 1e-4
    iteration_info = []

    # 初始合法化和优化
    two_rdm_legal, current_eps, current_E = legalized(experimental_2rdm, num_particles, one_body_coefficients, two_body_coefficients, verbose=True, max_iters=30000, accuracy=1e-7)
    
    print(f"current_eps_init is {current_eps}, and energy is {current_E }.")
    
    iteration_info.append({'d': d, 'current_eps_now': current_eps, 'current_eps': current_eps, 'E': current_E, 'E_thm': ground_state_energy - c, 'iterations': 0})

    while True:
        E, _, _, _, _ = optimize(two_rdm_legal, num_particles, one_body_coefficients, two_body_coefficients, epsilon=current_eps, accuracy=accuracy)
        current_eps_now = current_eps

        if E + c < ground_state_energy:
            if step_size <= threshold:
                break
            else:
                current_eps -= step_size
                step_size /= 10
        else:
            current_eps += step_size
            
        iteration_info.append({'d': d, 'current_eps_now': current_eps_now, 'current_eps': current_eps, 'E': E, 'E_thm': ground_state_energy - c, 'iterations': len(iteration_info)})
    
    iteration_info_df = pd.DataFrame(iteration_info)

    print(iteration_info_df)

    # 确保变量是字符串并且适合用于文件命名
    molecule = str(molecule)  # 如果molecule是其他类型，确保转换为字符串
    prob_1 = str(prob_1)  # 同样保证prob_1是字符串
    prob_2 = str(prob_2)  # 同理
    d = str(d)
    shots_num = str(shots_num)

    # 保存结果为 CSV 文件
    file_name_csv = f"iteration_info_eps_{molecule}_{prob_1}_{prob_2}_{d}_{shots_num}.csv"
    iteration_info_df.to_csv(file_name_csv, index=True)
    print("CSV文件已创建:", file_name_csv)

    # 保存结果为 pkl 文件
    file_path = f"iteration_info_eps_{molecule}_{prob_1}_{prob_2}_{d}_{shots_num}.pkl"  # 为pkl文件定义路径
    with open(file_path, 'wb') as file:
        pickle.dump(iteration_info_df, file)
    print("文件已创建:", file_path)

    return iteration_info_df['current_eps'].iloc[0], iteration_info_df['current_eps'].iloc[-1]

def calculate_single_point(molecule, d, accuracy, k, prob_1, prob_2, HEAp):
    """
    Calculates a single result point from scratch based on primary inputs.

    This function handles all data loading and processing for a single 'k' value.

    Args:
        molecule (str): The name of the molecule (e.g., 'H2').
        prob_1 (float): First probability parameter (currently unused in calculation but kept for signature consistency).
        prob_2 (float): Second probability parameter (currently unused in calculation but kept for signature consistency).
        d (float): The internuclear distance.
        shots_num (int): Number of shots (currently unused but kept for signature consistency).
        accuracy (float): Accuracy parameter for the optimizer.
        k (int): The multiple of delta to use for epsilon.

    Returns:
        dict: A dictionary containing the final calculation result for the given k.
        None: If any required file cannot be found or data is missing.
    """
    print(f"--- Starting calculation for {molecule} at d={d}, k={k} ---")
    current_dir = os.getcwd()
    d_str = f"{d:.1f}"

    try:
        # 1. Load Hamiltonian and RDM data
        hamiltonian_path = os.path.join(current_dir, f"{molecule}_{d}_sto-3g.pkl")
        with open(hamiltonian_path, 'rb') as file:
            H_data = pickle.load(file)

        one_rdm_path = os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_one_rdm_list.pkl')
        with open(one_rdm_path, 'rb') as f:
            one_rdm_list = pickle.load(f)

        two_rdm_path = os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_two_rdm_list.pkl')
        with open(two_rdm_path, 'rb') as f:
            two_rdm_list = pickle.load(f)

        # 2. Load the delta value
        delta_path = os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_df_2_rdm.pkl')
        with open(delta_path, 'rb') as f:
            delta_data = pickle.load(f)
            
        delta = delta_data["CIRC"]['WGF']

    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return None
    except KeyError as e:
        print(f"Error: A required key was not found in a data file. {e}")
        return None
    
    # 3. Extract and prepare data for calculation
    one_body_coefficients = H_data['one_body_coefficients']
    two_body_coefficients = np.einsum('ijkl->ijlk', H_data['two_body_coefficients']) # Pre-transpose
    constant = H_data['constant']
    num_particles = H_data['n_particles']
    experimental_2rdm = np.einsum('ijlk->ijkl', two_rdm_list[3]) # Transpose to match coefficients

    # 4. Perform the core calculation
    eps_CDR = k * delta
    print(f"Loaded delta={delta:.6f}, setting epsilon={eps_CDR:.6f}")
    
    E_thm, two_rdm_res, one_rdm_res, _, _ = legalize_then_optimize(
        experimental_2rdm,
        num_particles,
        one_body_coefficients,
        two_body_coefficients,
        accuracy=accuracy,
        epsilon=eps_CDR
    )

    # 5. Package and return the final result
    single_result = {
        'k': k,
        'd': d,
        'molecule': molecule,
        'delta': delta,
        'eps_CDR': eps_CDR,
        'E_thm_CDR': E_thm + constant,
        'one_rdm_res_ij_CDR': one_rdm_res,
        'two_rdm_res_ijkl_CDR': two_rdm_res
    }
    
    print(f"Calculation finished. E_thm_CDR = {single_result['E_thm_CDR']:.8f}")
    return single_result

def process_single_case(molecule, prob_1, prob_2, d, shots_num, accuracy, HEAp):
    
    print(f"Now finding optimal eps, parameters are : {prob_1},{prob_2},{d},{shots_num}")

    result_data = {}
    
    d_str = f"{d:.1f}" 
    current_dir = os.getcwd() # 获取当前工作目录的字符串 (e.g., '/home/user/project')
    input_filename = f"{molecule}_{d}_sto-3g.pkl"
    full_input_path = os.path.join(current_dir, input_filename) # 安全地拼接路径
    
    # 3. 在 open() 中使用构建好的完整路径
    print(f"Attempting to load file: {full_input_path}")
    try:
        with open(full_input_path, 'rb') as file:
            H2_d = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Input file not found. Skipping this distance.")
        print(f"Searched for file at: {full_input_path}")
        return f"Failed for d={d:.2f}"

    try:
        with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_one_rdm_list.pkl'), 'rb') as f:
            loaded_one_rdm_list = pickle.load(f)
        with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_two_rdm_list.pkl'), 'rb') as f:
            loaded_two_rdm_list = pickle.load(f)
         
        two_body_coefficients = H2_d['two_body_coefficients']
        one_body_coefficients = H2_d['one_body_coefficients']
        
        constant = H2_d['constant']
        FCI_val = H2_d['FCI_val']
        num_particles = H2_d['n_particles']
        Precise_diagonalization_energy = H2_d['Precise_diagonalization_energy']
        two_D_no_noisy = loaded_two_rdm_list[2]
        one_D_no_noisy = loaded_one_rdm_list[2]
        two_D_noisy = loaded_two_rdm_list[3]
        one_D_noisy = loaded_one_rdm_list[3]
        
        energy_no_noisy = np.sum(one_body_coefficients * one_D_no_noisy) + 0.5 * np.sum(two_body_coefficients * two_D_no_noisy) + constant
        
        energy_noisy = np.sum(one_body_coefficients * one_D_noisy) + 0.5 * np.sum(two_body_coefficients * two_D_noisy) + constant
        
        result_data['FCI_val'] = FCI_val
        result_data['energy_no_noisy'] = energy_no_noisy
        result_data['energy_noisy'] = energy_noisy
        
        print("++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++")
        
        two_body_coefficients = np.einsum('ijkl->ijlk', H2_d['two_body_coefficients'])
        experimental_2rdm_list = []
        # experimental_2rdm_list.append(loaded_two_rdm_list[3])
        experimental_2rdm_list.append(np.einsum('ijlk->ijkl', loaded_two_rdm_list[3]))
        
        print(experimental_2rdm_list[0].ndim)
        
        print("++++++++++++++++++++++++++++")

        eps_max = 10000
        
        E_thm_max, two_rdm_res_ijkl_max, one_rdm_res_ij_max, _, _ = legalize_then_optimize(experimental_2rdm_list[0], num_particles, one_body_coefficients, two_body_coefficients, accuracy=accuracy, epsilon = eps_max)
        print("++++++++++++++++++++++++++++") 
        print(E_thm_max)
        
        
        Convert_two_rdm = np.einsum('ijlk->ijkl', loaded_two_rdm_list[0])
        # print(two_rdm_res_ijkl_max)

        difference = Convert_two_rdm - two_rdm_res_ijkl_max

        # 计算差的 Frobenius 范数
        norm_difference = np.linalg.norm(difference)

        print("Norm of the difference:", norm_difference)
        
        
        energy = np.sum(one_body_coefficients * one_rdm_res_ij_max) + 0.5 * np.sum(two_body_coefficients * two_rdm_res_ijkl_max) + constant


        result_data['one_rdm_res_ij_max'] = one_rdm_res_ij_max
        result_data['two_rdm_res_ijkl_max'] = two_rdm_res_ijkl_max
        result_data['E_thm_max'] = E_thm_max + constant  
        
        
        print(energy)
        
        print(result_data['E_thm_max'])
        
        print("++++++++++++++++++++++++++++")
        
        current_eps_init, eps_optimal = find_optimal_eps(molecule, prob_1, prob_2, d, shots_num, 
            Precise_diagonalization_energy,
            constant,
            num_particles,
            accuracy,
            one_body_coefficients,
            two_body_coefficients,
            experimental_2rdm_list[0]
        )
        
        # 更新 result_data 字典
        result_data['leg_eps'] = current_eps_init
        result_data['eps_optimal'] = eps_optimal
        print("++++++++++++++++++++++++++++")
        
        print("--- Loading delta value from file ---")
        try:
            with open(os.path.join(current_dir, f'{molecule}_d{d_str}_{prob_1}_{prob_2}_{HEAp}_df_2_rdm.pkl'), 'rb') as f:
                loaded_df_2_rdm = pickle.load(f)
            
            delta = loaded_df_2_rdm["CIRC"]['WGF']
            print(f"Successfully loaded delta = {delta:.6f}")
            print(f"The threshold value current_eps_init = {current_eps_init:.6f}")

            # 2. 初始化循环变量
            k = 1  # delta 的倍数，从1开始
            cdr_results_list = [] # 创建一个列表来存储每次循环的结果

            print("\n--- Searching for the smallest k and running one calculation ---")

            while True: # 启动一个无限循环，直到某个条件满足
                eps_CDR = k * delta
    
                # 每次循环，都执行核心计算和数据保存
                print(f"\nProcessing k={k}:")
                print(f"Setting epsilon for legalize_then_optimize to {eps_CDR:.6f}")

                # 运行核心计算函数
                E_thm_CDR, two_rdm_res_ijkl_CDR, one_rdm_res_ij_CDR, G_ijkl_CDR, Q_ijkl_CDR = legalize_then_optimize(
                    experimental_2rdm_list[0], 
                    num_particles, 
                    one_body_coefficients, 
                    two_body_coefficients, 
                    accuracy=accuracy, 
                    epsilon=eps_CDR
                )

                # 保存本次循环的结果
                final_result = {
                    'k': k,
                    'eps_CDR': eps_CDR,
                    'E_thm_CDR': E_thm_CDR + constant,
                    'one_rdm_res_ij_CDR': one_rdm_res_ij_CDR,
                    'two_rdm_res_ijkl_CDR': two_rdm_res_ijkl_CDR
                    }
                cdr_results_list.append(final_result)
    
                print(f"Calculation finished. E_thm_CDR = {final_result['E_thm_CDR']:.8f}")

                # 检查是否满足退出条件
                if eps_CDR > eps_optimal:
                    # 如果条件满足，我们已经保存了这次的结果，现在可以安全地退出了
                    print(f"\n✅ 退出条件满足于 k={k}。循环结束。")
                    print(f"(k * delta) = {eps_CDR:.6f} > {eps_optimal:.6f}")
                    break
    
                # 如果条件不满足，k 自增并继续下一次循环
                k += 1

            # 循环结束后的最终检查
            print(f"\nLoop finished. Final check for k={k}:")
            print(f"Condition: (k * delta) = {k * delta:.6f} < {current_eps_init:.6f} (False)")

            # 4. 将存储了所有循环结果的列表存入主 result_data 字典
            result_data['cdr_iterations'] = cdr_results_list
            
        except FileNotFoundError:
            print(f"Error: Could not find the file to load delta. Skipping the CDR loop.")
        except KeyError:
            print(f"Error: Keys 'CIRC' or 'THM' not found in the loaded data. Skipping the CDR loop.")

        print("++++++++++++++++++++++++++++") 
        # 初始化 E_thm_CF
  
        return result_data, k , delta, eps_optimal
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    return None

# --- 整理后的 main 函数 ---
if __name__ == "__main__":
    
    # 1. 设置参数列表
    molecule = 'H2'
    basis = "sto-3g"
    
    shots_num = 2**15
    accuracy = 1e-7
    
    d_list = []
    current_value = 0.4
    while current_value <= 1.2:
        d_list.append(round(current_value, 1))
        current_value += 0.1
    while current_value <= 3.5:
        d_list.append(round(current_value, 1))
        current_value += 0.3

    print(d_list)
    
    prob_1_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    prob_2_list = [0.01, 0.015, 0.02]
    HEAp_list = [2, 4, 6, 8]
    
    '''    
    d_list = [0.4]
    prob_1_list = [0.001]
    prob_2_list = [0.01]
    HEAp_list = [2, 4]
    '''
    # 2. 使用 itertools.product 创建所有参数的组合
    param_combinations = list(itertools.product(d_list, prob_1_list, prob_2_list, HEAp_list))
    
    print(f"总共要运行 {len(param_combinations)} 种参数组合。")
    print("参数组合示例:", param_combinations[:3]) # 打印前3个组合看看
    
    def run_all_for_one_combo(params):
        # 从元组中解包参数
        d, prob_1, prob_2, HEAp = params
        
        # 第一个计算
        replaced_gates = process_distance(d, molecule, prob_1, prob_2, shots_num, HEAp)
        
        # 第二个计算
        result_data, k_opt, delta, eps_opt = process_single_case(molecule, prob_1, prob_2, d, shots_num, accuracy, HEAp)
        
        # (可选) 保存详细的 pickle 文件
        # filename = f"data_result_{molecule}_d{d}_p1{prob_1}_p2{prob_2}_h{HEAp}.pkl"
        # with open(filename, 'wb') as file:
        #     pickle.dump(result_data, file)
            
        # 将所有输入参数和输出结果打包成一个字典返回
        return {
            'distance': d,
            'prob_1': prob_1,
            'prob_2': prob_2,
            'HEAp': HEAp,
            'replaced_gates_num': replaced_gates,
            'k_optimal': k_opt,
            'delta': delta,
            'eps_optimal': eps_opt
        }
        
    # 4. 并行运行所有参数组合
    print("\n开始并行处理所有参数组合...")
    current_time_start = datetime.datetime.now()
    
    # joblib 会返回一个字典列表，每个字典都是一次运行的结果
    results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_all_for_one_combo)(params) for params in param_combinations
    )
    
    current_time_end = datetime.datetime.now()
    print(f"--- 所有任务完成! 总耗时: {current_time_end - current_time_start} ---")

    # 5. 将结果列表转换为 DataFrame 并保存
    print("\n正在将所有结果整合到汇总文件中...")
    
    df_summary = pd.DataFrame(results_list)

    # 保存为 CSV 文件
    output_filename = f"simulation_summary_{molecule}_{basis}_multi_param.csv"
    df_summary.to_csv(output_filename, index=False)

    print(f"\n所有汇总数据已保存至 {output_filename}")
    print("\n数据预览 (前5行):")
    print(df_summary.head())


'''
if __name__ == "__main__":

    molecule = 'H2'
    basis= "sto-3g"
    d_list = [0.4]

    d_list = []
    current_value = 0.4
    while current_value <= 1.2:
        d_list.append(round(current_value, 1))
        current_value += 0.1
    while current_value <= 3.5:
        d_list.append(round(current_value, 1))
        current_value += 0.3

    print(d_list)
    # 噪声模型的参数（在循环外定义一次即可）
    prob_1 = 0.005 # 1-比特门的噪声概率
    prob_2 = 0.02 # 2-比特门的噪声概率
    HEAp = 2
    shots_num = 2 ** 15
    # num_particles = 4  
    accuracy = 1e-7
    # 使用 joblib 并行执行
    # n_jobs=-1 表示使用所有可用的CPU核心
    # verbose=10 会打印详细的进度信息，非常有用！

    print("\nStarting parallel processing with joblib...")
    replaced_gates_num = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_distance)(d, molecule, prob_1, prob_2, shots_num, HEAp) for d in d_list
        )

    print("replaced_gates_num:", replaced_gates_num)

    print("\n--- All jobs completed! ---")

    current_time_start = datetime.datetime.now()

    print(f"Now starting the execution of VQE + VRDM, time is: {current_time_start}")

    k = 3

    for d in d_list:
    # 调用处理函数
        result_data, k_optimal , delta, eps_optimal= process_single_case(molecule, prob_1, prob_2, d, shots_num, accuracy)
        single_result = calculate_single_point(molecule, d, accuracy, k)
        filename = f"data_result_{molecule}_{d}_{basis}.pkl"
    # 使用pickle保存字典
    with open(filename, 'wb') as file:
        pickle.dump(result_data, file)

    print(f"All data are saved as {filename}")

    current_time_end = datetime.datetime.now()

    print(f"Now ending VRDM, Total time is: {current_time_end - current_time_start}")
    
    data_row = {
        'distance': [current_d],
        'prob_1': [prob_1],
        'prob_2': [prob_2],
        'HEAp': [HEAp],
        'shots_num': [shots_num],
        'accuracy': [accuracy],
        'k_optimal': [k_optimal],
        'delta': [delta],
        'eps_optimal': [eps_optimal],
        'replaced_gates_num': [replaced_gates_num]
    }

    # 2. Convert the dictionary to a pandas DataFrame
    df_row = pd.DataFrame(data_row)

    # 3. Define the output filename
    output_filename = "simulation_summary.csv"

    # 4. Save the data to the CSV file
    df_row.to_csv(output_filename, index=False)

    print(f"Single row of data saved to {output_filename}")
    print("\nData preview:")
    print(df_row)
'''