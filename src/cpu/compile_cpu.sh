#!/bin/bash
  
# MIT License - Copyright (c) 2025 Hongyi Guan
# See LICENSE file for full license text

export CPLUS_INCLUDE_PATH=/path/to/eigen:$CPLUS_INCLUDE_PATH

g++ -O3 -std=c++17 -fopenmp compare_bicg_adi.cpp -o compare_bicg_adi

