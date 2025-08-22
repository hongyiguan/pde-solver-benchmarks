/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */

/*
 * The Alternating Directions Implicit (ADI) Solver
 * for a 2D linear equation (1-a*D^2) m = f.
 * When a is small, this can be approximately decomposed to
 * A = (1-a*D^2) = (1-a*(Dx^2+Dy^2)) = (1-a*Dx^2)(1-a*Dy^2) + o(a^2)
 * Therefore, the equation Am = f is solved as 
 * Ax.u = f, Ay.m = u, 
 * where Ax and Ay are tridiagonal matrices,
 * easily solved by Gauss elimination (Thomas algorithm)
 */

#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <random>


// 1D Laplacian operator with Neumann boundary condition
std::vector<std::vector<double>> lap_1D_pure(int N = 256) {
    std::vector<std::vector<double>> L(N, std::vector<double>(N, 0));
    
    L[0][1] = L[N - 2][N - 1] = 2;

    for (int i = 0; i < N; ++i) {
        L[i][i] = -2;
        if (i > 0) {
            L[i][i - 1] = 1;
        }

        if (i < N - 1) {
            L[i][i + 1] = 1;
        }
    }

    return L;
}

// 1D Laplacian operator with Neumann boundary condition 
// + a defect in the center (identity matrix)
std::vector<std::vector<double>> lap_1D_def(int N = 256, int inner_square = 32) {
    int n = N / 2 - inner_square / 2;
    auto D = lap_1D_pure(n);

    std::vector<std::vector<double>> result(N, std::vector<double>(N, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = D[i][j];
            result[i + n + inner_square][j + n + inner_square] = D[i][j];
        }
    }

    for (int i = n; i < n + inner_square; ++i) {
        result[i][i] = 1;
    }

    return result;
}

struct Diagonals {
	std::vector<double> a, b, c;
};

Diagonals extract_diagonals(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    Diagonals diag;
    diag.a = std::vector<double>(n);
    diag.b = std::vector<double>(n);
    diag.c = std::vector<double>(n);

    for (int i = 0; i < n; ++i) {
        diag.b[i] = A[i][i];
        if (i > 0) {
            diag.a[i] = A[i][i - 1];
        }
        if (i < n - 1) {
            diag.c[i] = A[i][i + 1];
        }
    }

    return diag;
}

// Thomas algorithm for tridiagonal systems - O(n)
std::vector<double> thomas(const Diagonals& diag, const std::vector<double>& f) {
    int n = f.size();
    std::vector<double> d = f, b = diag.b;
    std::vector<double> x(n);

    // Forward pass
    for (int i = 1; i < n; ++i) {
        double w = diag.a[i] / b[i - 1];
        b[i] -= w * diag.c[i - 1];
        d[i] -= w * d[i - 1];
    }

    // Backward substitution
    x[n - 1] = d[n - 1] / b[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = (d[i] - diag.c[i] * x[i + 1]) / b[i];
    }

    return x;
}

std::vector<std::vector<double>> identity(int n) {
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

std::vector<std::vector<double>> subtractMatrix(const std::vector<std::vector<double>>& A, double alpha, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    std::vector<std::vector<double>> result(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = A[i][j] - alpha * B[i][j];
        }
    }
    return result;
}

struct ADISystem {
    int N, inner_start, inner_end;
    double alpha;
    Diagonals pure, def;
};    

ADISystem build_ADI_system(int N, int inner_square, double alpha) {
    ADISystem sys;
    sys.N = N;
    sys.inner_start = N / 2 - inner_square / 2;
    sys.inner_end = N / 2 + inner_square / 2;

    auto L_pure = lap_1D_pure(N);
    auto L_def = lap_1D_def(N, inner_square);
    auto I = identity(N);
    auto A_pure = subtractMatrix(I, alpha, L_pure);
    auto A_def = subtractMatrix(I, alpha, L_def);
    sys.pure = extract_diagonals(A_pure);
    sys.def = extract_diagonals(A_def);
    return sys;
}

// The Alternating Directions Implicit (ADI) Solver 
std::vector<double> ADI(const ADISystem& sys, const std::vector<std::vector<double>>& f_vec) {
    int N = sys.N;
    auto diag_pure = sys.pure;
    auto diag_def = sys.def;

    std::vector<std::vector<double>> u(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> x(N, std::vector<double>(N, 0));
    std::vector<double> x_final(N * N);
    int inner_start = sys.inner_start;
    int inner_end = sys.inner_end;

    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        auto columnVec = std::vector<double>(N);
        for (int i = 0; i < N; ++i) {
            columnVec[i] = f_vec[i][j];
        }
	std::vector<double> solution;
        if (inner_start <= j && j < inner_end){
		solution = thomas(diag_def, columnVec);
	}else{
		solution = thomas(diag_pure, columnVec);
	}
        for (int i = 0; i < N; ++i) {
            u[i][j] = solution[i];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
	std::vector<double> solution;
        if (inner_start <= i && i < inner_end){
		solution = thomas(diag_def, u[i]);
        }else{
		solution = thomas(diag_pure, u[i]);
        }
        for (int j = 0; j < N; ++j) {
            x[i][j] = solution[j];
            x_final[i * N + j] = x[i][j];
        }
    }

    return x_final;
}

double randomDouble(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int main() {
    const int N = 1024;
    const int innerSquare = N / 4;
    const int innerStart = N / 2 - innerSquare / 2;
    const int innerEnd = N / 2 + innerSquare / 2;
    const double alpha = 0.0243;

    std::vector<std::vector<double>> f(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (innerStart <= i && i < innerEnd && innerStart <= j && j < innerEnd) {
                f[i][j] = 0;
            } else {
                f[i][j] = randomDouble(0, 5);
            }
        }
    }

    ADISystem sys;
    sys = build_ADI_system(N, innerSquare, alpha);
    double totalTime = 0;
    int iterations = 10;

    for (int it = 0; it < iterations; ++it) {
        auto result = ADI(sys, f);
    }

    //double averageTime = totalTime / iterations;
    //std::cout << "Average time taken over " << iterations << " iterations: " << averageTime << " seconds" << std::endl;

    return 0;
}
