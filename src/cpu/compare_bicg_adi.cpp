/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#define main eigen_bicg_dummy_main
#include "eigen_bicg.cpp"
#undef main

#define main adi_dummy_main
#include "adi.cpp"
#undef main

using std::cout;
using std::endl;

inline int idx(int i, int j, int N) { return i * N + j; }

static std::vector<std::vector<double>> to2D(const Eigen::VectorXd& v, int N) {
    std::vector<std::vector<double>> M(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[i][j] = v[idx(i, j, N)];
    return M;
}

static Eigen::VectorXd toEigen(const std::vector<double>& v) {
    Eigen::VectorXd x(v.size());
    for (int k = 0; k < (int)v.size(); ++k) x[k] = v[k];
    return x;
}

int main(int argc, char** argv) {
    int N = (argc >= 2) ? std::atoi(argv[1]) : 1024;
    int innerSquare = (argc >= 3) ? std::atoi(argv[2]) : N/4;
    double alpha = (argc >= 4) ? std::stod(argv[3]) : 0.04;
    std::string fname = (argc >= 5) ? argv[4] : "1024_256_f.bin"; 
    const int N2 = N * N;
    // Solve the linear system 10 times and calculate the average time
    int count = 10;

    try {
	// Method 1: Solve with Eigen's bicgstab + ILU preconditioner
        Eigen::SparseMatrix<double> L = generateLaplacian(N, innerSquare);

        Eigen::SparseMatrix<double> I(N2, N2);
        I.setIdentity();

        Eigen::SparseMatrix<double> A = I - alpha * L;

        Eigen::VectorXd f = readVectorFromFile(fname, N);

        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
	solver.setTolerance(1e-8);
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "BiCGSTAB: factorization failed.\n";
            return 1;
        }

	double avg_time_bicg = 0;
	Eigen::VectorXd x_bicg;

	for (int i = 0; i < count; i++){
        auto t0 = std::chrono::high_resolution_clock::now();
        x_bicg = solver.solveWithGuess(f, f);
        auto t1 = std::chrono::high_resolution_clock::now();
	double time_bicg = std::chrono::duration<double>(t1 - t0).count();
	avg_time_bicg += time_bicg / count;
	}

        if (solver.info() != Eigen::Success) {
            std::cerr << "BiCGSTAB: solve failed.\n";
            return 1;
        }

        // Method 2: Solve with ADI
	auto f2D = to2D(f, N);

	ADISystem sys;
	sys = build_ADI_system(N, innerSquare, alpha);

	double avg_time_adi = 0;
	std::vector<double> x_adi_vec;

	for (int i = 0; i < count; i++){
        auto t2 = std::chrono::high_resolution_clock::now();
        x_adi_vec = ADI(sys, f2D);
        auto t3 = std::chrono::high_resolution_clock::now();
	double time_adi = std::chrono::duration<double>(t3 - t2).count();
	avg_time_adi += time_adi / count;
	}

        Eigen::VectorXd x_adi = toEigen(x_adi_vec);

        const Eigen::VectorXd diff = x_bicg - x_adi;

        const double norm2_ref = x_bicg.norm();
        const double rel_l2 = diff.norm() / (norm2_ref > 0 ? norm2_ref : 1.0);

        const double normInf_ref = x_bicg.lpNorm<Eigen::Infinity>();
        const double rel_inf =
            diff.lpNorm<Eigen::Infinity>() / (normInf_ref > 0 ? normInf_ref : 1.0);

        const double eps = 1e-12;
        double acc = 0.0;
        for (int k = 0; k < N2; ++k) {
            const double denom = std::abs(x_bicg[k]) + eps;
            acc += std::abs(diff[k]) / denom;
        }
        const double mean_rel_elem = acc / double(N2);

        cout << std::fixed << std::setprecision(6);
        cout << "Problem size: N=" << N << " (grid " << N << "x" << N << "), innerSquare=" << innerSquare
             << ", alpha=" << alpha << "\n";
        cout << "RHS file: " << fname << "\n\n";

        cout << "BiCGSTAB solve time: " << avg_time_bicg << " s\n";
        cout << "ADI solve time: " << avg_time_adi << " s\n\n";

        cout << "Agreement metrics (x_bicg vs. x_adi):\n";
        cout << "  Relative L2:      " << rel_l2 << "\n";
        cout << "  Relative L_infty: " << rel_inf << "\n";
        cout << "  Mean |dx|/|x|:    " << mean_rel_elem << "\n";

        cout << "\nChecksums:\n";
        cout << "  sum(x_bicg) = " << x_bicg.sum() << "\n";
        cout << "  sum(x_adi)   = " << x_adi.sum() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
