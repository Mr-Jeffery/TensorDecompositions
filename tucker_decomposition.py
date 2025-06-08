import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from scipy.linalg import svd
from tensorly.random import random_tucker
from tensorly.tenalg import mode_dot # type: ignore
from tensorly.tucker_tensor import tucker_to_tensor

'''
Tucker Tensor Decomposition
This Python script implements Tucker decomposition (also known as Higher-Order SVD), 
a tensor decomposition method that factorizes a tensor into a core tensor multiplied 
by factor matrices along each mode.

Some features:
    Implementation of Tucker decomposition using SVD-based approach
    Automatic rank determination based on singular value cutoff
    Reconstruction error evaluation
    
Three unit tests demonstrating different use cases:
    Synthetic tensor (3rd-order)
    Synthetic tensor (4th-order)
    Random tensor (3rd-order)
'''

# Evaluate the reconstruction error (relative error) of Tucker decomposition
def recon_error_eval(core: np.ndarray, factor: list[np.ndarray], tensor: np.ndarray) -> float:
    recon_tensor = tucker_to_tensor((core, factor))
    rel_error = tl.norm(recon_tensor - tensor) / tl.norm(tensor)
    return rel_error

# Tucker decomposition (High-order SVD)
def tucker_decomposition(
    tensor: np.ndarray, 
    rank: list[int], 
    cutoff: float
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    shape = tensor.shape  # Tensor shape [n1, n2, ..., nd]
    order = len(shape)    # order d
    
    # Compute the factor matrix A(1), A(2), ..., A(d) by SVD
    factor = []
    output_rank = []
    for n in range(order):
        # X(n): Matricization of the input tensor along different modes
        X = tl.base.unfold(tensor, n) 
        
        # Singular value decomposition and truncation
        U, S, Vh = svd(X)  # SVD of the unfolded matrix
        truncS = S[np.where(S > cutoff)] # Truncate singular values by cutoff
        trunc_rank = len(truncS)  # Truncation dimension (rank)
        
        # A(n) <- R_n leading left singular vectors of X(n)
        if (trunc_rank > rank[n]):
            A = U[:, 0:rank[n]]
            output_rank.append(rank[n])
        else:
            A = U[:, 0:trunc_rank]
            output_rank.append(trunc_rank)
        factor.append(A)
            
    # Compute the core tensor
    # core = tensor *_1 A(1)T *_2 A(2)T ... *_d A(d)T  
    core = mode_dot(tensor, factor[0], 0, transpose = True)
    for n in range(1, order):
        core = mode_dot(core, factor[n], n, transpose = True)

    return core, factor, output_rank

def tucker_unit_test(
    shape: list[int],
    rank: list[int] | None,
    input_rank: list[int],
    cutoff: float,
    seed_value: int = 0,
    random_tensor: bool = False
) -> None:
    """
    Generalized unit test for Tucker decomposition.

    Args:
        shape: Shape of the tensor.
        rank: Tucker rank for generating synthetic tensor.
        input_rank: Input rank for decomposition.
        cutoff: Singular value cutoff.
        random_state: Random seed.
        random_tensor: If True, generate a random tensor instead of a synthetic Tucker tensor.
    """
    print("Tucker unit test starts with input:")
    print(f"Shape = {shape}, Rank = {rank}, Input rank = {input_rank}, Cutoff = {cutoff}, Seed value = {seed_value}, Random tensor = {random_tensor}")
    start_t = tm.time()

    if random_tensor:
        if seed_value is not None:
            seed(seed_value)
        tensor = rand(*shape)
    else:
        tucker_factor = random_tucker(shape, rank, random_state=seed_value)
        tensor = tucker_to_tensor(tucker_factor)

    core, factor, output_rank = tucker_decomposition(tensor, input_rank, cutoff)
    recon_error = recon_error_eval(core, factor, tensor)

    end_t = tm.time()
    print(f"Tucker unit test ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTucker factor rank = {output_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return

# Example calls replacing the previous unit tests:
print("Unit test 1 starts!")
tucker_unit_test(
    shape=[20, 30, 20],
    rank=[15, 30, 18],
    input_rank=[20, 40, 20],
    cutoff=1e-10,
    seed_value=10
)

print("Unit test 2 starts!")
tucker_unit_test(
    shape=[20, 50, 10, 30],
    rank=[15, 30, 7, 25],
    input_rank=[50, 50, 50, 50],
    cutoff=1e-10,
    seed_value=20
)

print("Unit test 3 starts!")
tucker_unit_test(
    shape=[50, 60, 100],
    rank=None,  # Not used for random tensor
    input_rank=[100, 100, 100],
    cutoff=1e-10,
    seed_value=20,
    random_tensor=True
)

