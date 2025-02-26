import time
import numpy as np
import sparse as sp
import tensorly as tl
import tensorly.contrib.sparse as stl
from tensorly.base import tensor_to_vec, vec_to_tensor
from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac # The dense CP
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac # The sparse CP

def superdiagonal_tensor(shape, value=1.0):
    if len(shape) != 3:
        raise ValueError("Shape must be 3-dimensional for a 3-order tensor")
    
    tensor = np.zeros(shape)  # Create empty tensor
    min_dim = min(shape)      # Find the minimum dimension
    
    for i in range(min_dim): 
        tensor[i, i, i] = value  # Fill the super-diagonal
    
    return tensor

def unit_test_1():
    print("Unit test 1 starts!")
    # CP factors
    shape = (100, 101, 102)
    seed = (1, 2, 3)
    rank = 5
    starting_weights = stl.ones((rank))
    starting_factors = [sp.random((shape[i], rank), random_state=seed[i]) for i in range(3)]
    print(type(starting_factors[0]))

    # CP to tensor
    tensor = cp_to_tensor((starting_weights, starting_factors))
    print(type(tensor))

    # Dense CP decomposition
    t = time.time()
    dense_cp = parafac(tensor, 5, init='random')
    print(time.time() - t)

    # Sparse CP decomposition
    t = time.time(); 
    sparse_cp = sparse_parafac(tensor, 5, init='random'); 
    print(time.time() - t)

def unit_test_2():
    print("Unit test 2 starts!")
    # Shape and density of the sparse tensor
    shape = (7, 10, 9)
    density = 0.3
   
    # Generate the sparse tensor in by hard threshold
    np.random.seed(1)
    tensor = np.random.random(shape)
    randl = np.random.random(shape)
    tensor = np.where(randl > density, 0, tensor)
    
    # Convert dense format to COO format
    tensor = sp.COO(tensor)
    print(type(tensor))
    
    # Sparse CP (which is extremely slow)
    #t = time.time()
    #sparse_cp = sparse_parafac(tensor, 5, init='random'); 
    #print(f"Sparse CP takes {time.time() - t} seconds")
    
    # Dense CP
    tensor_full = tensor.todense()
    rank = 5
    t = time.time()
    dense_cp = parafac(tensor_full, rank, init='random')
    print(f"Dense CP takes {time.time() - t} seconds")
    
    # Kronecker product of CP factors A B C     
    M = np.kron(np.kron(dense_cp.factors[0], dense_cp.factors[1]), dense_cp.factors[2])
    
    # Compute the least square problem by pseudo inverse
    T_vec = tensor_to_vec(tensor_full)
    pinvM = np.linalg.pinv(M)
    G_vec = pinvM @ T_vec
    
    # Core consistency
    diagT = superdiagonal_tensor([rank, rank, rank], 1.0)
    G_ten =  vec_to_tensor(G_vec, [rank, rank, rank])
    temp = np.sum((G_ten - diagT)**2)
    cc = 100 * (1 - temp / rank)
    print(cc)

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
    
    # Kronecker product of CP factors A B C     
    M = np.kron(np.kron(dense_cp.factors[0], dense_cp.factors[1]), dense_cp.factors[2])
    
    # Compute the least square problem by pseudo inverse
    T_vec = tensor_to_vec(tensor)
    pinvM = np.linalg.pinv(M)
    G_vec = pinvM @ T_vec
    
    # Core consistency
    diagT = superdiagonal_tensor([rank, rank, rank], 1.0)
    G_ten =  vec_to_tensor(G_vec, [rank, rank, rank])
    temp = np.sum((G_ten - diagT)**2)
    cc = 100 * (1 - temp / rank)
    print(cc)
    
    


#unit_test_1()
#unit_test_2()
unit_test_3()

