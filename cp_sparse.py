import time 
import numpy as np
import sparse as sp
import tensorly as tl
import tensorly.contrib.sparse as stl
from tensorly.base import tensor_to_vec, vec_to_tensor
from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac # The dense CP
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac # The sparse CP

# superdiagonal_tensor (used by core consistency diagnostic) generates a super-diagonal tensor
# in which all elements are 0 except for those at position (i, i, i) = value.
def superdiagonal_tensor(shape, value=1.0):
    # Only works for 3d tensor
    if len(shape) != 3:
        raise ValueError("Shape must be 3-dimensional for a 3-order tensor")
    
    # Create a tensor
    tensor = np.zeros(shape)  # Create empty tensor
    min_dim = min(shape)      # Find the minimum dimension
    
    # Assign value to the diagonal
    for i in range(min_dim): 
        tensor[i, i, i] = value  # Fill the super-diagonal
    
    return tensor

# Core consistency diagnostic
def core_consistency(tensor, cp_factor):
    # Only works for 3d tensor
    if len(cp_factor.factors) != 3:
        raise ValueError("Must be 3 CP factors for a 3-order tensor")
    
    # Kronecker product of CP factors A B C     
    M = np.kron(np.kron(cp_factor.factors[0], cp_factor.factors[1]), cp_factor.factors[2])
    
    # Compute the least square problem by pseudo inverse
    T_vec = tensor_to_vec(tensor)  # Vectorization
    pinvM = np.linalg.pinv(M)  # pseudo-inverse which is used to solve the least square problem 
    G_vec = pinvM @ T_vec
    
    # Core consistency
    rank = cp_factor.factors[0].shape[1]  # Rank is the column number of every CP factor
    diagT = superdiagonal_tensor([rank, rank, rank], 1.0)  # Create a super-diagonal tensor
    G_ten =  vec_to_tensor(G_vec, [rank, rank, rank])  # Vector->Tensor
    temp = np.sum((G_ten - diagT)**2)
    cc = 100 * (1 - temp / rank)
    
    return cc

# Unit Test 1&2: These two tests try to decompose a dense tensor and a sparse tensor separately,
# showing that the sparse CP decomposition implemented in TensorLy is significantly slower than 
# the dense CP decomposition.   
# As a result, performing CP decomposition in tensorly in a sparse style is not very practical.  
# I recommend keeping using the dense CP decomposition (tensorly.decomposition.parafac) 
# even for sparse data, as long as the data can be fit into memory.
def unit_test_1():
    print("Unit test 1 starts!")
    # Random CP factors
    shape = (8, 10, 6)
    rank = 5
    np.random.seed(20)
    starting_weights = stl.ones((rank))
    starting_factors = [np.random.rand(shape[i], rank) for i in range(3)]

    # CP to tensor
    tensor = cp_to_tensor((starting_weights, starting_factors))

    # Dense CP decomposition
    t = time.time()
    dense_cp = parafac(tensor, 5, init='random')
    print(f"Dense CP takes {time.time() - t} seconds.")

    # Sparse CP decomposition
    tensor = sp.COO(tensor)
    t = time.time(); 
    sparse_cp = sparse_parafac(tensor, 5, init='random'); 
    print(f"Sparse CP takes {time.time() - t} seconds.")

    print("Unit test 1 ends!\n") 
    return

# Unit test 2: test dense/sparse CP of a sparse tensor
def unit_test_2():
    print("Unit test 2 starts!")
    # Shape and density of the sparse tensor
    shape = (20, 20, 20)
    density = 0.05
   
    # Generate the sparse tensor in by hard threshold
    np.random.seed(1)
    tensor = np.random.random(shape)
    randl = np.random.random(shape)
    tensor = np.where(randl > density, 0, tensor)
    
    # Convert dense format to COO format
    tensor = sp.COO(tensor)
    rank = 5
    
    # Sparse CP (which is extremely slow)
    t = time.time()
    sparse_cp = sparse_parafac(tensor, rank, init='random'); 
    print(f"Sparse CP takes {time.time() - t} seconds")
    
    # Dense CP
    tensor_full = tensor.todense()
    t = time.time()
    dense_cp = parafac(tensor_full, rank, init='random')
    print(f"Dense CP takes {time.time() - t} seconds")
    
    print("Unit test 2 ends!\n")
    return
    
# Unit test 3 gives a simple demonstration of how to use core consistency diagnostic 
# When rank = 1, the core consistency should be 100.0
def unit_test_3():
    print("Unit test 3 starts!")
    # CP factors
    shape = (3, 3, 3)
    rank = 3
    np.random.seed(20)
    starting_weights = stl.ones((rank))
    starting_factors = [np.random.rand(shape[i], rank) for i in range(3)]

    # CP to tensor
    tensor = cp_to_tensor((starting_weights, starting_factors))

    # Dense CP decomposition
    rank = 1
    t = time.time()
    dense_cp = parafac(tensor, rank, init='random')
    print(f"Dense CP took {time.time() - t} seconds.")
    
    reconT = cp_to_tensor(dense_cp)
    print(f"Recon error = {tl.norm(reconT-tensor,2)}")
    
    cc = core_consistency(tensor, dense_cp)
    print(f"Core consistency = {cc}")

    print("Unit test 3 ends!\n")
    return

# The `cp_cc_workflow` function takes a data file path, data shape, and CP rank as input.  
# It performs CP decomposition on the data loaded from the given path,  
# computes the core consistency, and returns the result.  
def cp_cc_workflow(filename, dataShape, rank):
    # Load data and reshape
    print("Loading data...")
    data = np.fromfile(filename, dtype=np.float32)
    tensor = data.reshape(dataShape) # CESM 2D data shape is 1800*3600 for example

    # Dense CP
    print("CP decomposition starts...")
    t = time.time()
    dense_cp = parafac(tensor, rank, init='random')
    print(f"Dense CP took {time.time() - t} seconds.")

    # Sparse CP
    #sparse_thres = 1e-6
    #tensor = sp.COO.from_numpy(tensor * (np.abs(tensor) > sparse_thres))
    #t = time.time()
    #sparse_cp = sparse_parafac(tensor, rank, init='random')
    #print(f"Dense CP took {time.time() - t} seconds.")
    
    print("Computing core consistency...")
    cc = core_consistency(tensor, dense_cp)
    print(f"cc = {cc}")
    return cc
    
#unit_test_1()
#unit_test_2()
#unit_test_3()

filename = "/home/zmeng5/SDRBENCH-CESM-ATM-26x1800x3600/CLDICE_1_26_1800_3600.f32"
cc = cp_cc_workflow(filename, (26, 1800, 3600), 5)