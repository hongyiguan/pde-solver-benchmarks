#!/bin/bash

# MIT License - Copyright (c) 2025 Hongyi Guan
# See LICENSE file for full license text

# Run script for PETSc CUDA examples
# 
# Prerequisites:
# - Install PETSc with CUDA support enabled
# - Set PETSC_DIR environment variable to your PETSc installation directory
# - Ensure CUDA and MPI are available in your system

if [ -n "$PETSC_ARCH" ] && [ -d "$PETSC_DIR/$PETSC_ARCH/lib" ]; then
  export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="$PETSC_DIR/lib:${LD_LIBRARY_PATH}"
fi

ILU_EXE=./petsc_cuda_bicg_ilu
AMG_EXE=./petsc_cuda_gmres_amg

declare -a PROBLEMS=(
  "1024 256 ../f_vec/1024_256_f.bin"
  "2048 512 ../f_vec/2048_512_f.bin"
  "512 128 ../f_vec/512_128_f.bin"
  "256 64  ../f_vec/256_64_f.bin"
)

declare -a ALPHAS=(0.01 0.02 0.04)

REPEATS=10

for prob in "${PROBLEMS[@]}"; do
  set -- $prob
  N=$1
  INNER=$2
  FPATH=$3

  for alpha in "${ALPHAS[@]}"; do
    echo "============================================"
    echo " BICG + ILU | N=$N inner=$INNER alpha=$alpha"
    echo "============================================"
    mpirun -np 1 $ILU_EXE \
      -N $N -inner $INNER -alpha $alpha -fpath $FPATH -repeats $REPEATS \
      -vec_type cuda -mat_type aijcusparse \
      -ksp_type bcgs -pc_type ilu -pc_factor_levels 0 \
      -ksp_rtol 1e-8

    echo "============================================="
    echo " GMRES + AMG | N=$N inner=$INNER alpha=$alpha"
    echo "============================================="
    mpirun -np 1 $AMG_EXE \
      -N $N -inner $INNER -alpha $alpha -fpath $FPATH -repeats $REPEATS \
      -ksp_rtol 1e-8
  done
done

