# Third-Party Notices

This project depends on the following external libraries:

- **Eigen** — MPL 2.0 license

  - Home: https://eigen.tuxfamily.org
  - Usage: Header-only C++ library for CPU solvers.
- **PETSc** — BSD 2-Clause license

  - Home: https://petsc.org
  - Usage: GPU solvers (PETSc with CUDA support).
- **MPI (e.g., OpenMPI)** — BSD-style license

  - Home: https://www.open-mpi.org/
  - Usage: Required dependency for PETSc. Not called directly by this code.
- **CUDA Toolkit** — NVIDIA EULA (proprietary)

  - Home: https://developer.nvidia.com/cuda-downloads
  - Usage: Required for PETSc GPU backend. Not called directly by this code.
