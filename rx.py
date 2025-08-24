import os, ctypes
os.add_dll_directory(r"C:\Github\AMGX\build\Release")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
ctypes.CDLL(r"C:\\Github\\AMGX\\build\\Release\\amgxsh.dll")

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import pyamgx

pyamgx.initialize()

# Initialize config and resources:
# cfg = pyamgx.Config().create_from_dict({
#    "config_version": 2,
#         "determinism_flag": 1,
#         "exception_handling" : 1,
#         "solver": {
#             "monitor_residual": 1,
#             "solver": "BICGSTAB",
#             "convergence": "RELATIVE_INI_CORE",
#             "preconditioner": {
#                 "solver": "NOSOLVER"
#         }
#     }
# })

cfg = pyamgx.Config().create_from_file(r"C:\\Github\\AMGX\\src\\configs\\FGMRES_NOPREC.json")

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:
M = sparse.csr_matrix(np.random.rand(5, 5))
rhs = np.random.rand(5)
sol = np.zeros(5, dtype=np.float64)

A.upload_CSR(M)
b.upload(rhs)
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(sol)
print("pyamgx solution: ", sol)
print("scipy solution: ", splinalg.spsolve(M, rhs))

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
