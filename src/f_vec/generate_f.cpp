/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */

/*
 * Example: ./generate_f 512 128 /path/to/your/f_file
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    int N = (argc >= 2) ? std::atoi(argv[1]) : 512;
    int innerSquare = (argc >= 3) ? std::atoi(argv[2]) : (N / 4);
    std::string outfile = (argc >= 4) ? argv[3] : ("f_N" + std::to_string(N) + ".bin");

    unsigned long long seed = 1919810;

    if (N <= 0 || innerSquare < 0 || innerSquare > N) {
        std::cerr << "Invalid N or innerSquare\n";
        return 1;
    }

    const int N2 = N * N;
    std::vector<double> f(N2);

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 5.0);

    for (int i = 0; i < N2; ++i) f[i] = dist(gen);

    const int innerStart = N / 2 - innerSquare / 2;
    const int innerEnd   = innerStart + innerSquare;
    auto idx = [N](int i, int j) { return i * N + j; };

    if (innerSquare > 0) {
        for (int i = innerStart; i < innerEnd; ++i) {
            for (int j = innerStart; j < innerEnd; ++j) {
                f[idx(i, j)] = 0.0;
            }
        }
    }

    std::ofstream ofs(outfile, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << outfile << "\n";
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(N2 * sizeof(double)));
    if (!ofs) {
        std::cerr << "Failed to write data.\n";
        return 1;
    }
    ofs.close();

    std::cout << "Wrote " << N2 << " doubles to " << outfile
              << " (N=" << N << ", innerSquare=" << innerSquare
              << ")\n";
    return 0;
}

