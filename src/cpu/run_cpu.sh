#!/bin/bash

# MIT License - Copyright (c) 2025 Hongyi Guan
# See LICENSE file for full license text

ulimit -s unlimited

export OMP_NUM_THREADS=48

CPU_EXE=./compare_bicg_adi

declare -a PROBLEMS=(
  "1024 256 ../f_vec/1024_256_f.bin"
  "2048 512 ../f_vec/2048_512_f.bin"
  "512 128 ../f_vec/512_128_f.bin"
  "256 64 ../f_vec/256_64_f.bin"
)

declare -a ALPHAS=(0.01 0.02 0.04)

for prob in "${PROBLEMS[@]}"; do
  set -- $prob
  N=$1
  INNER=$2
  FPATH=$3

  for alpha in "${ALPHAS[@]}"; do
    $CPU_EXE $N $INNER $alpha $FPATH
  done
done

