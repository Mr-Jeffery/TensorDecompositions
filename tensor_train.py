import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from numpy.linalg import svd
from tensorly.random import random_tt
from tensorly.tt_tensor import tt_to_tensor

'''
Tensor Train (TT) Decomposition
This Python script implements Tensor Train decomposition, a method for representing 
high-dimensional tensors as a sequence of three-order tensors. The implementation uses 
TT-SVD approach with adaptive rank selection.

Features
Implementation of TT decomposition using SVD with adaptive rank selection
Automatic rank determination based on truncation parameter
Reconstruction error evaluation

Three unit tests demonstrating different use cases:
Synthetic tensor (3rd-order)
Synthetic tensor (4th-order)
Random tensor (4th-order)
'''

# Evaluate the reconstruction error (relative error) of Tucker decomposition
def recon_error_eval(tt_factor: list[tl.tensor], tensor: np.ndarray) -> float:
    recon_tensor = tt_to_tensor(tt_factor)
    rel_error = tl.norm(recon_tensor - tensor) / tl.norm(tensor) 
    return rel_error

def tensor_train_decomposition(tensorX: tl.tensor, r_max: int, eps: float, verbose: int = 0) -> list[tl.tensor]:
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
    delta = (eps / np.sqrt(dim - 1)) * tl.norm(tensorX, 2)  # Truncation parameter
    
    W = tensorX        # Copy tensor X -> W
    nbar = W.size      # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing tt factors
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    
    for i in iterlist:
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])  # Reshape W
        U, S, Vh = svd(W)  # SVD of W matrix
        # Compute rank r
        s = 0
        j = S.size 
    
        while s <= delta * delta:  # r_delta_i = min(j:sigma_j+1^2 + sigma_j+2^2 + ... <= delta^2)
            j -= 1
            s += S[j] * S[j]
            if j == 0:
                break
        j += 1
        ri = min(j, r_max)  # r_i-1 = min(r_max, r_delta_i)
    
        if verbose == 1:
            approxLR = U[:, 0:ri] @ np.diag(S[0:ri]) @ Vh[0:ri, :]
            rerror = tl.norm(approxLR - W, 2) / tl.norm(W, 2)
            print(f"Iteration {i} -- low rank approximation error = {rerror}")
    
        Ti = tl.reshape(Vh[0:ri, :], [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri  # Renewal r
        W = U[:, 0:ri] @ np.diag(S[0:ri])  # W = U[..] * S[..]
        ttList.append(Ti)  # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList

def tensor_train_unit_test(
    shape: list[int],
    rank: list[int] | None = None,
    r_max: int = 50,
    cutoff: float = 1e-10,
    seed_val: int = 10,
    random_tensor: bool = False
) -> None:
    """
    Run a tensor train decomposition unit test with configurable parameters.

    Parameters:
        shape (list): Shape of the tensor.
        rank (list or None): TT rank for synthetic TT tensor. Ignored if random_tensor=True.
        r_max (int): Maximum TT rank for decomposition.
        cutoff (float): Truncation parameter (epsilon).
        seed_val (int): Random seed.
        random_tensor (bool): If True, use a completely random tensor; otherwise, use a synthetic TT tensor.
    """
    print(f"Tensor train starts with input:")
    print(f"Shape = {shape}, Rank = {rank}, r_max = {r_max}, Cutoff = {cutoff}, Seed value = {seed_val}, Random tensor = {random_tensor}")
    start_t = tm.time()

    if random_tensor:
        seed(seed_val)
        tensor = rand(*shape)
        tensor_tl = tl.tensor(tensor)
    else:
        synthetic_tt = random_tt(shape, rank, random_state=seed_val)
        tensor_tl = tt_to_tensor(synthetic_tt)

    tensor_tl = np.asarray(tensor_tl)  # Ensure ndarray
    tt_factor = tensor_train_decomposition(tensor_tl, r_max, cutoff)
    recon_error = recon_error_eval(tt_factor, tensor_tl)
    tt_rank = [fac.shape[2] for fac in tt_factor[:-1]]

    end_t = tm.time()
    print(f"Unit test ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTensor-train rank = {tt_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return


# Example usages:
print("Unit test 1 starts!")
tensor_train_unit_test(
    shape=[50, 50, 50], 
    rank=[1, 30, 30, 1], 
    r_max=50, 
    cutoff=1e-10, 
    seed_val=10, 
    random_tensor=False
)

print("Unit test 2 starts!")
tensor_train_unit_test(
    shape=[30, 40, 20, 30], 
    rank=[1, 20, 30, 13, 1], 
    r_max=50, 
    cutoff=1e-10, 
    seed_val=10, 
    random_tensor=False
)

print("Unit test 3 starts!")
tensor_train_unit_test(
    shape=[10, 20, 30, 10], 
    rank=None, 
    r_max=200, 
    cutoff=1e-10, 
    seed_val=20, 
    random_tensor=True
)