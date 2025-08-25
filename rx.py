import os, ctypes
os.add_dll_directory(r"C:\Github\AMGX\build\Release")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
ctypes.CDLL(r"C:\\Github\\AMGX\\build\\Release\\amgxsh.dll")

import numpy as np
import scipy.sparse as sparse
from scipy.io import mmread
import scipy.sparse.linalg as splinalg

import pyamgx

pyamgx.initialize()

A_path = "C:\\Github\\RXMesh\\output\\MCF_matrices\\Nefertiti_A.mtx"
rhs_path = "C:\\Github\\RXMesh\\output\\MCF_matrices\\Nefertiti_B.mtx"

cfg = pyamgx.Config().create_from_file("rx.json")

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b0 = pyamgx.Vector().create(rsc)
b1 = pyamgx.Vector().create(rsc)
b2 = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg, 'dDDI')

# Read the matrices 
N = 1000
#M = sparse.csr_matrix(np.random.rand(N, N), dtype=np.float64)
M = mmread(A_path).tocsr()
rhs = mmread(rhs_path) 
rhs = np.asarray(rhs)
sol = np.zeros(rhs.shape[0])

print(f"Matrix shape {M.shape} with {M.nnz} NNZ")
print(f"RHS shape {rhs.shape}")

# Upload the matrices 
A.upload_CSR(M)
b0.upload(rhs[:,0])
b1.upload(rhs[:,1])
b2.upload(rhs[:,2])
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b0, x)
solver.solve(b1, x)
solver.solve(b2, x)

# Download solution
#x.download(sol)
print(f"pyamgx took {solver.iterations_number} iter")
print(f"pyamgx solver status: {solver.status}")
#print("scipy solution: ", splinalg.spsolve(M, rhs))

# Clean up:
A.destroy()
x.destroy()
b0.destroy()
b1.destroy()
b2.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
