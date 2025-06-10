import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from scipy.linalg import svd
from tensorly.random import random_tucker
from tensorly.tenalg import mode_dot # type: ignore
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.decomposition import tucker as tucker_decomposition_tl

from typing import Optional, List

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
    input_rank: list[int],
    gen_rank: Optional[List[int]] = None,
    cutoff: Optional[float] = None,
    tol: float = 1e-10,
    n_iter_max: int = 1000,
    seed_value: int = 0,
) -> None:
    """
    Generalized unit test for Tucker decomposition. Two modes of tensor generation:
    1. Synthetic Tucker tensor with specified shape and rank.
    2. Random tensor with specified shape.
    Two modes of decomposition:
    1. SVD-based Tucker decomposition with a cutoff for singular values.
    2. TensorLy-based Tucker decomposition with a tolerance for convergence.

    Args:
        shape: Shape of the tensor.
        input_rank: Input rank for decomposition.
        gen_rank: Tucker rank for generating synthetic tensor.
        cutoff: Cutoff for singular values in Tucker decomposition, used in cut-off mode only.
        tol: the algorithm stops when the variation in the reconstruction error is less than the tolerance, 
              used in TensorLy mode only.
        n_iter_max: Maximum number of iterations for convergence, used in TensorLy mode only.
        seed_value: Seed value for random number generation.
        random_tensor: If True, generate a random tensor instead of a synthetic Tucker tensor.
    """
    print("Tucker unit test starts with input:")
    print(f"Shape = {shape}, Input rank = {input_rank}, Generation Rank = {gen_rank}, Cutoff = {cutoff}, "
          f"Tolerance = {tol}, Seed value = {seed_value}")
    start_t = tm.time()

    if not gen_rank:
        seed(seed_value)
        tensor = rand(*shape)
    else:
        tucker_factor = random_tucker(shape, gen_rank, random_state=seed_value)
        tensor = tucker_to_tensor(tucker_factor)

    if cutoff :
        core, factor, output_rank = tucker_decomposition(tensor, input_rank, cutoff)
        recon_error = recon_error_eval(core, factor, tensor)
    else:
        output_tucker, recon_error = tucker_decomposition_tl(tensor=tensor, rank=input_rank, tol=tol,n_iter_max=n_iter_max,return_errors=True)
        recon_error = recon_error[-1]  # Get the last error value
        output_rank = output_tucker.rank

    end_t = tm.time()
    print(f"Tucker unit test ends! It took {end_t - start_t} seconds")
    print(f"Tensor shape = {shape},\nTucker factor rank = {output_rank},")
    print(f"Reconstruction error = {recon_error}\n")
    return

# Example calls replacing the previous unit tests:
print("Unit test 1 starts!")
tucker_unit_test(
    shape=[20, 30, 20],
    gen_rank=[15, 30, 18],
    input_rank=[15, 30, 18],
)

print("Unit test 2 starts!")
tucker_unit_test(
    shape=[20, 50, 10, 30],
    gen_rank=[15, 30, 7, 25],
    input_rank=[50, 50, 50, 50],
    seed_value=20,
    cutoff=1e-12,
)

# Test of a completely random tensor
print("Unit test 3 starts!")
tucker_unit_test(
    shape=[50, 60, 100],
    input_rank=[100, 100, 100],
    tol=1e-12,
    seed_value=20,
    cutoff=1e-12,
)

