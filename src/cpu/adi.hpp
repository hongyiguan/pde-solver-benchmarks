/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */


#pragma once
#include <vector>
#include <utility>

struct Diagonals { std::vector<double> a, b, c; };
struct ADISystem {
    int N, inner_start, inner_end;
    double alpha;
    Diagonals pure, def;
};

ADISystem build_ADI_system(int N, int inner_square, double alpha);

std::pair<double, std::vector<double>>
ADI(const ADISystem& sys, const std::vector<std::vector<double>>& f_vec);
