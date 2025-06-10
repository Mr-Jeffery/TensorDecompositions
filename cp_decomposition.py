import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from tensorly.random import random_cp
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac as cp_decomposition
from typing import Optional

'''
One of the greatest features of tensors is that they can be represented 
compactly in decomposed forms and we have powerful methods with guarantees 
to obtain these decompositions.

This Python script demonstrates one type of tensor decomposition: Canonical 
Polyadic Decomposition (also known as CANDECOMP/PARAFAC, CP, or PARAFAC decomposition). 
The idea is to express a tensor as a sum of rank-one tensors, which are outer products of vectors.

This demo uses TensorLy's parafac function rather than my own implementation. 
Before running the script, please ensure you have TensorLy installed in your environment.
Three unit tests demonstrating different use cases:
    work_example: A binary tensor (matrix, 2nd-order)
    unit_test_1: Random tensor (3rd-order)
    unit_test_2: Random tensor (4th-order)
'''

# A work example from tensorly
def work_example() -> None:
    print("Work example starts!")
    # The input tensor (here the tensor is a two-dimensional matrix) 
    tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]) 
    
    # CP decomposition. Input: tensor, rank 
    # ( A rank-r CP decomposes a tensor into a linear combination of r rank-1 tensors )
    cp_result = cp_decomposition(tensor, rank=2)
    
    print(f"number of factors (matrices) = {len(cp_result.factors)}")
    print(f"Shape of factors: {[f.shape for f in cp_result.factors]}")
    
    # Reconstruct CP factor to tensor
    recon_tensor = tl.cp_to_tensor(cp_result)

    # Evaluate the reconstruction error 
    error = tl.norm(recon_tensor - tensor) / tl.norm(tensor) 
    print(f"Reconstruction error = {error}")
    print("Work example ends!")
    return

def cp_unit_test(
    shape: list[int], 
    input_rank: int,
    gen_rank: Optional[int] = None,
    tol: float = 1e-10,
    seed_value : int = 0, 
)-> None:
    print(f"CP decomposition starts with input:")
    print(f"Shape = {shape}, Input rank = {input_rank}, Generation Rank = {gen_rank}, Tolerance = {tol}, Seed value = {seed_value}\n")

    start_t = tm.time()

    if gen_rank is None:
        seed(seed_value)
        tensor = rand(*shape)
    else:
        cp_factor = random_cp(shape, rank=gen_rank, random_state=seed_value)
        tensor = cp_to_tensor(cp_factor)
    
    cp_result = cp_decomposition(tensor=tensor, rank=input_rank, tol=tol, return_errors=True)
    assert isinstance(cp_result, tuple), "The output should be a tuple containing the CP decomposition and reconstruction error."
    output_cp, recon_error = cp_result
    
    end_t = tm.time()
    print(f"CP unit test ends! It took {end_t - start_t} seconds")
    print(f"number of factors (matrices) = {len(output_cp.factors)}")
    print(f"Shape of factors: {output_cp.shape}")
    print(f"Reconstruction error = {recon_error[-1]}\n")
    return


work_example()

# Example usage:
print("Unit test 1 starts!")
cp_unit_test(
    shape=[20, 10, 30], 
    seed_value=20, 
    input_rank=200,
    gen_rank=200,
    tol=1e-12,
)

print("Unit test 2 starts!")
cp_unit_test(
    shape=[20, 30, 20, 10], 
    seed_value=10, 
    input_rank=300,
)

