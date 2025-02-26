# Tensor Decomposition Methods

This repository contains implementations of three fundamental tensor decomposition methods: Canonical Polyadic (CP), Tucker, and Tensor Train (TT) decompositions. Each method provides a different way to decompose high-dimensional tensors into more manageable components.

## Python Package Requirements
- numpy
- scipy
- tensorly

## Implemented Methods

### 1. CP Decomposition (`cp_decomposition.py`)
Function _parafac_ decomposes a tensor into a linear combination of r rank-1 tensors.

**Input Arguments:**
- `tensor`: Input tensor to decompose, could be either tensorly object or numpy ndarray.
- `rank`: Number of rank-one tensors in the decomposition. **A parameter to be tuned**. Generally, a greater CP rank leads to a lower reconstruction error.

**Output:**
- CP decomposition factors (list of matrices)

**Reconstruction from outputs to the tensor:**
- tensorly.cp_to_tensor

### 2. Tucker Decomposition (`tucker_decomposition.py`)
Function _tucker_decomposition_ decomposes a tensor into a core tensor and factor matrices.

**Input Arguments:**
- `tensor`: Input tensor to decompose.  
- `rank`: List of maximum ranks for each mode. **A parameter to be tuned**. Generally, a greater rank along each mode leads to a lower reconstruction error.
- `cutoff`: Singular value truncation threshold. **A parameter to be tuned**. Generally, a smaller cutoff leads to a lower reconstruction error.

**Output:**
- Core tensor
- Factor matrices (list)
- Output ranks for each mode

**Reconstruction from outputs to the tensor:**
- tensorly.tucker_tensor.tucker_to_tensor

### 3. Tensor Train Decomposition (`tensor_train.py`)
Function _tensor_train_decomposition_ decomposes a tensor into a sequence of interconnected three-dimensional tensors.

**Input Arguments:**
- `tensorX`: Input tensor to decompose
- `r_max`: Maximum allowed TT-rank. **A parameter to be tuned**. Generally, a greater TT rank leads to a lower reconstruction error.
- `eps`: Truncation parameter. **A parameter to be tuned**. Generally, a smaller cutoff leads to a lower reconstruction error.
- `verbose`: Verbosity flag (0 or 1). 0 by default.

**Output:**
- List of TT-cores (three-dimensional tensors)

**Reconstruction from outputs to the tensor:**
- tensorly.tt_tensor.tt_to_tensor

## Unit Tests
Each script includes multiple unit tests demonstrating:
- Decomposition of random tensors
- Reconstruction error evaluation

## Usage
Each script can be run independently:
```bash
python cp_decomposition.py
python tucker_decomposition.py
python tensor_train.py
```

The scripts will execute their respective unit tests and display:
- Shape of input tensor
- (Output)Ranks of the decomposition
- Reconstruction error
- Computation time
