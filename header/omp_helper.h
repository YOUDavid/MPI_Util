#ifndef OMP_HELPER_H
#define OMP_HELPER_H
#define _GNU_SOURCE
#include "shared_types.h"
#include "shared_library.h"
#include <omp.h>
void omp_dgemm(const charptr RESTRICT tra_a, const charptr RESTRICT tra_b,
               const intptr RESTRICT m_arr, const intptr RESTRICT n_arr, const intptr RESTRICT k_arr,
               const doubleptr RESTRICT alpha, const doubleptrptr RESTRICT a_arr, const intptr RESTRICT lda,
               const doubleptrptr RESTRICT b_arr, const intptr RESTRICT ldb, const doubleptr RESTRICT beta,
               doubleptrptr RESTRICT c_arr, const intptr  RESTRICT ldc,
               const RESTRICT intptr completed, const int num);


int omp_sum_int(const intptr RESTRICT arr, const int num);

void omp_arr_mul_int(intptr RESTRICT c, const intptr RESTRICT a, const intptr RESTRICT b, const int num);

void omp_arr_ofs_dou(doubleptrptr RESTRICT arr, const intptr RESTRICT ofs, const int num);

void omp_print(int len, double* arr);

void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta);
             
int stick_to_core(const int core_id);             
#endif
