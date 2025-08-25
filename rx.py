import os, ctypes
os.add_dll_directory(r"C:\Github\AMGX\build\Release")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
ctypes.CDLL(r"C:\\Github\\AMGX\\build\\Release\\amgxsh.dll")

import numpy as np
import scipy.sparse as sparse
from scipy.io import mmread
import scipy.sparse.linalg as splinalg

import pyamgx

import igl

pyamgx.initialize()

model_name = "sphere1"
write_output = True

A_path = f"C:\\Github\\RXMesh\\output\\MCF_matrices\\{model_name}_A.mtx"
rhs_path = f"C:\\Github\\RXMesh\\output\\MCF_matrices\\{model_name}_B.mtx"

cfg = pyamgx.Config().create_from_file("rx.json")

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = [pyamgx.Vector().create(rsc), pyamgx.Vector().create(rsc), pyamgx.Vector().create(rsc)]
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg, 'dDDI')

# Read the matrices 
M = mmread(A_path).tocsr()
rhs = mmread(rhs_path) 
rhs = np.asarray(rhs)

print(rhs)

rows = rhs.shape[0]
cols = rhs.shape[1]

sol = np.zeros(rows)

if write_output:
    output = np.zeros(rhs.shape)

print(f"Matrix shape {M.shape} with {M.nnz} NNZ")
print(f"RHS shape {rhs.shape}")

# Upload the matrices 
A.upload_CSR(M)
for i in range(cols):
    b[i].upload(rhs[:,i])

x.upload(sol)

# Setup and solve system:
solver.setup(A)

for i in range(cols):    
    solver.solve(b[i], x)
    if write_output:
        x.download(sol)
        output[:,i] = sol

# Download solution
#x.download(sol)
print(f"pyamgx took {solver.iterations_number} iter")
print(f"pyamgx solver status: {solver.status}")

if write_output:
    v, f = igl.read_triangle_mesh(f"C:\\Github\\RXMesh\\output\\MCF_matrices\\{model_name}.obj")
    igl.write_triangle_mesh(f"{model_name}_amgx.obj", output, f)

scipy_output = splinalg.spsolve(M, rhs)
igl.write_triangle_mesh(f"{model_name}_scipy.obj", scipy_output, f)

# Clean up:
A.destroy()
x.destroy()
for i in range(cols):
    b[i].destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
