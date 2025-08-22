/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>


using namespace Eigen;
using namespace std;

// Generate Eigen Sparse matrix for 2D Laplacian operator
// with a defect in the center and Neumann boundary condition
Eigen::SparseMatrix<double> generateLaplacian(int N = 256, int innerSquare = 32) {
    const int N2 = N * N;
    Eigen::SparseMatrix<double> L(N2, N2);
    L.reserve(VectorXi::Constant(N2, 5)); // Reserve space for 5 non-zeros per row, heuristic

    const int innerStart = N / 2 - innerSquare / 2;
    const int innerEnd = N / 2 + innerSquare / 2;

    auto idx = [N](int i, int j) { return i * N + j; };

    auto insertOnce = [&](int row,int col){
	if (L.coeff(row,col) == 0.0)
	    L.insert(row,col) = 1.0;
    };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int id = idx(i, j);
	    bool inDefectx = innerStart <= i && i < innerEnd;
	    bool inDefecty = innerStart <= j && j < innerEnd;
	    bool inDefect = inDefectx && inDefecty;
            if (inDefect) {
                L.insert(id, id) = 1.0;
            } else {
	    L.insert(id,id) = -4.0;
	    bool leftNeumann = (i == 0 || (i == innerEnd && inDefecty));
	    bool rightNeumann = (i == N-1 || (i == innerStart-1 && inDefecty));
	    bool downNeumann = (j == 0 || (j == innerEnd && inDefectx ));
	    bool upNeumann = (j == N-1 || (j == innerStart-1 && inDefectx));

	    if (leftNeumann) L.coeffRef(id, idx(i+1, j)) = 2.0; else insertOnce(id, idx(i-1, j));
	    if (rightNeumann) L.coeffRef(id, idx(i-1, j)) = 2.0; else insertOnce(id, idx(i+1, j));
	    if (downNeumann) L.coeffRef(id, idx(i, j+1)) = 2.0; else insertOnce(id, idx(i, j-1));
	    if (upNeumann) L.coeffRef(id, idx(i, j-1)) = 2.0; else insertOnce(id, idx(i, j+1));
	    }
        }
    }
    L.makeCompressed();
    return L;
}

Eigen::VectorXd readVectorFromFile(const std::string &filename, int N) {
    const size_t N2 = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> buffer(N2);

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    ifs.read(reinterpret_cast<char*>(buffer.data()),
             static_cast<std::streamsize>(N2 * sizeof(double)));
    if (!ifs) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return Eigen::Map<Eigen::VectorXd>(buffer.data(), N2);
}

int main() {
    const int N = 1024;          
    const int innerSquare = 256;
    const int N2 = N * N;
    double alpha = 0.04;

    Eigen::SparseMatrix<double> L = generateLaplacian(N, innerSquare);
    Eigen::SparseMatrix<double> I(N2, N2);
    I.setIdentity();

    Eigen::SparseMatrix<double> A = I - alpha * L;

    Eigen::VectorXd f = readVectorFromFile("1024_256_f.bin", N);

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
    solver.compute(A);

    double avg_time = 0.0;
    for (int i = 0; i < 10; i++) {
        auto start = chrono::high_resolution_clock::now();

        if (solver.info() != Success) {
            cout << "Decomposition failed" << endl;
            return -1;
        }

        VectorXd x = solver.solveWithGuess(f, f);
        if (solver.info() != Success) {
            cout << "Solving failed" << endl;
            return -1;
        }

        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;

        cout << "Time taken to solve Ax = f: " << elapsed.count() << " seconds." << endl;
        avg_time += elapsed.count() / 10.0;
    }

    cout << "\n>> Average time taken to solve Ax = f: "
         << avg_time << " seconds." << endl;

    return 0;
}

