#!/bin/bash

# MIT License - Copyright (c) 2025 Hongyi Guan
# See LICENSE file for full license text

# Build script for PETSc CUDA examples
# 
# Prerequisites:
# - Install PETSc with CUDA support enabled
# - Set PETSC_DIR environment variable to your PETSc installation directory
# - Ensure CUDA and MPI are available in your system

if [ -z "$PETSC_DIR" ]; then
    echo "Error: PETSC_DIR environment variable is not set"
    echo "Please set it to your PETSc installation directory"
    exit 1
fi

export PETSC_ARCH=""

echo "Compiling petsc_cuda_bicg_ilu.cpp..."
mpicxx -O3 -std=c++17 petsc_cuda_bicg_ilu.cpp -o petsc_cuda_bicg_ilu \
  -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
  -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc

echo "Compiling petsc_cuda_gmres_amg.cpp..."
mpicxx -O3 -std=c++17 petsc_cuda_gmres_amg.cpp -o petsc_cuda_gmres_amg \
  -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
  -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc
