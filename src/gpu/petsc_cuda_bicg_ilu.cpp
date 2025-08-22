/*
 * MIT License - Copyright (c) 2025 Hongyi Guan
 * See LICENSE file for full license text
 */

#include <petscksp.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>

#ifndef PetscBoolCast
  #define PetscBoolCast(a) ((a) ? PETSC_TRUE : PETSC_FALSE)
#endif

static inline PetscInt idx(PetscInt N, PetscInt i, PetscInt j) { return i * N + j; }

// Assemble the same 2D Laplace operator as cpu/eigen_bicg.cpp
PetscErrorCode BuildLaplacian(Mat L, PetscInt N, PetscInt innerSquare)
{
  PetscFunctionBeginUser;
  const PetscInt N2 = N * N;
  const PetscInt innerStart = N/2 - innerSquare/2;
  const PetscInt innerEnd   = N/2 + innerSquare/2;

  for (PetscInt i = 0; i < N; ++i) {
    for (PetscInt j = 0; j < N; ++j) {
      const PetscInt r = idx(N,i,j);
      const PetscBool inDefectx = PetscBoolCast(innerStart <= i && i < innerEnd);
      const PetscBool inDefecty = PetscBoolCast(innerStart <= j && j < innerEnd);
      const PetscBool inDefect  = PetscBoolCast(inDefectx && inDefecty);

      if (inDefect) {
        PetscCall(MatSetValue(L, r, r, 1.0, INSERT_VALUES));
      } else {
        PetscCall(MatSetValue(L, r, r, -4.0, INSERT_VALUES));

        const PetscBool leftNeumann  = PetscBoolCast(i == 0 || (i == innerEnd   && inDefecty));
        const PetscBool rightNeumann = PetscBoolCast(i == N-1 || (i == innerStart-1 && inDefecty));
        const PetscBool downNeumann  = PetscBoolCast(j == 0 || (j == innerEnd   && inDefectx));
        const PetscBool upNeumann    = PetscBoolCast(j == N-1 || (j == innerStart-1 && inDefectx));

        if (leftNeumann)  PetscCall(MatSetValue(L, r, idx(N,i+1,j), 2.0, INSERT_VALUES));
        else              PetscCall(MatSetValue(L, r, idx(N,i-1,j), 1.0, INSERT_VALUES));

        if (rightNeumann) PetscCall(MatSetValue(L, r, idx(N,i-1,j), 2.0, INSERT_VALUES));
        else              PetscCall(MatSetValue(L, r, idx(N,i+1,j), 1.0, INSERT_VALUES));

        if (downNeumann)  PetscCall(MatSetValue(L, r, idx(N,i,j+1), 2.0, INSERT_VALUES));
        else              PetscCall(MatSetValue(L, r, idx(N,i,j-1), 1.0, INSERT_VALUES));

        if (upNeumann)    PetscCall(MatSetValue(L, r, idx(N,i,j-1), 2.0, INSERT_VALUES));
        else              PetscCall(MatSetValue(L, r, idx(N,i,j+1), 1.0, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L,   MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

// Read f vector binary file
PetscErrorCode LoadVecRaw(const std::string& path, Vec v)
{
  PetscFunctionBeginUser;
  PetscInt Nloc, Nglob;
  PetscCall(VecGetLocalSize(v, &Nloc));
  PetscCall(VecGetSize(v,       &Nglob));
  if (Nloc != Nglob) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This example expects a single-process run (Nloc==Nglob).");

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file");

  PetscScalar *arr = nullptr;
  PetscCall(VecGetArray(v, &arr));
  ifs.read(reinterpret_cast<char*>(arr), static_cast<std::streamsize>(sizeof(PetscScalar) * Nglob));
  if (!ifs) {
    PetscCall(VecRestoreArray(v, &arr));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Short read or error reading f file");
  }
  PetscCall(VecRestoreArray(v, &arr));
  PetscFunctionReturn(0);
}

int main(int argc, char** argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

  PetscInt  N          = 1024;           
  PetscInt  inner      = 256;           
  PetscReal alpha      = 0.04;      
  char      fpath[PETSC_MAX_PATH_LEN] = "../1024_256_f.bin";
  PetscInt  repeats    = 10;

  PetscOptionsBegin(PETSC_COMM_SELF, "", "Options", "");
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-N",      &N,      nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-inner",  &inner,  nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr, "-alpha",  &alpha,  nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-fpath", fpath, sizeof(fpath), nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-repeats",&repeats,nullptr));
  PetscOptionsEnd();

  const PetscInt N2 = N * N;

  // --- Create L on host, preallocate 5 nnz/row (matches stencil) ---
  Mat L;
  PetscCall(MatCreate(PETSC_COMM_SELF, &L));
  PetscCall(MatSetSizes(L, N2, N2, N2, N2));
  PetscCall(MatSetFromOptions(L));                      // allow -mat_type aijcusparse
  PetscCall(MatSeqAIJSetPreallocation(L, 5, nullptr));
  PetscCall(MatSetUp(L));

  PetscCall(BuildLaplacian(L, N, inner));

  // --- Build A = I - alpha * L ---
  Mat A;
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A));
  PetscCall(MatScale(A, -alpha));
  PetscCall(MatShift(A, 1.0)); 

  // --- Create vectors (on GPU if -vec_type cuda is used) ---
  Vec f, x;
  PetscCall(VecCreate(PETSC_COMM_SELF, &f));
  PetscCall(VecSetSizes(f, N2, N2));
  PetscCall(VecSetFromOptions(f));
  PetscCall(LoadVecRaw(std::string(fpath), f));

  PetscCall(VecDuplicate(f, &x));
  PetscCall(VecCopy(f, x));
  KSP ksp;
  PC  pc;
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPSetType(ksp, KSPBCGS));         // BiCGSTAB
  PetscCall(PCSetType(pc, PCILU));             // ILU (cuSPARSE-backed on GPU for AIJ CUSPARSE)
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));           // let CLI override types/tols

  // Warm-up (sets up GPU data structures & ILU factors once)
  PetscCall(KSPSolve(ksp, f, x));
  KSPConvergedReason reason;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (reason < 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Warm-up solve diverged (reason=%d)\n", (int)reason));
    PetscCall(PetscFinalize());
    return 1;
  }

  double avg = 0.0;
  for (PetscInt it = 0; it < repeats; ++it) {
    PetscCall(VecCopy(f, x));
    PetscReal t0, t1;
    PetscCall(PetscTime(&t0));
    PetscCall(KSPSolve(ksp, f, x));
    PetscCall(PetscTime(&t1));
    const double dt = (double)(t1 - t0);
    avg += dt / repeats;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Solve %d: %g s\n", (int)it+1, dt));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n>> Average time over %d runs: %g s\n", (int)repeats, avg));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&L));
  PetscCall(PetscFinalize());
  return 0;
}

