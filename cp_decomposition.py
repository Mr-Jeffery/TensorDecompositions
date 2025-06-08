import numpy as np
import time as tm
import tensorly as tl

from numpy.random import rand
from numpy.random import seed
from tensorly.decomposition import parafac

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
    cp_result = parafac(tensor, rank=2)
    
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
    seed_value : int, 
    rank: int
)-> None:
    print(f"CP decomposition starts with input:")
    print(f"Shape = {shape}, Seed value = {seed_value}, Rank = {rank}")
    start_t = tm.time()
    seed(seed_value)  # Set random seed
    tensor = rand(*shape)
    
    cp_result = parafac(tensor, rank)
    
    # Reconstruct CP factor to tensor
    recon_tensor = tl.cp_to_tensor(cp_result)

    # Evaluate the reconstruction error 
    error = tl.norm(recon_tensor - tensor) / tl.norm(tensor) 
    
    end_t = tm.time()
    print(f"CP unit test ends! It took {end_t - start_t} seconds")
    print(f"number of factors (matrices) = {len(cp_result.factors)}")
    print(f"Shape of factors: {[f.shape for f in cp_result.factors]}")
    print(f"Reconstruction error = {error}")
    return


work_example()

# Example usage:
print("Unit test 1 starts!")
cp_unit_test(
    shape=[20, 10, 30], 
    seed_value=20, 
    rank=200
)

print("Unit test 2 starts!")
cp_unit_test(
    shape=[20, 30, 20, 10], 
    seed_value=10, 
    rank=300
)

