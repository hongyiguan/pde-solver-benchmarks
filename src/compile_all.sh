#!/bin/bash

# MIT License - Copyright (c) 2025 Hongyi Guan
# See LICENSE file for full license text

cd cpu && bash ./compile_cpu.sh && cd ..
cd gpu && bash ./compile_gpu.sh && cd ..
cd f_vec && bash ./compile_f.sh && cd ..
echo "All compilations completed."