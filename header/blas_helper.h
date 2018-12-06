#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#include "shared_types.h"
#include "shared_library.h"
#include <omp.h>
void generate_supermatrix(NP_MATRIX **old, int i, int j, int k, int l, double **matrices, int *blockdim, int ndim, char *tras);

void delete_np_matrix(NP_MATRIX **np_matrix);

NP_MATRIX *new_np_matrix(const int m, const int n);

NP_MATRIX *cnew_np_matrix(const int m, const int n);

NP_MATRIX *renew_np_matrix(NP_MATRIX **old, const int m, const int n);

void strided_copy_np_matrix(NP_MATRIX *des, const int drow, const int dcol, NP_MATRIX *src, const int realm, const int realn, const int srow, const int scol);

void wrap_np_matrix(NP_MATRIX *wrapper, double *raw, const int m, const int n, const char tra);

void transpose_np_matrix(NP_MATRIX *old);

void flip_ij(NP_MATRIX **old, int i, int j, NP_MATRIX *matrix, int *blockdim);

void copy_np_matrix(NP_MATRIX *des, const int row, const int col, NP_MATRIX *src);

double index_np_matrix(NP_MATRIX *matrix, const int i, const int j);

void axpy_np_matrix(NP_MATRIX **result, NP_MATRIX *x, const double a, NP_MATRIX *y);

void scal_np_matrix(NP_MATRIX **result, NP_MATRIX *x, const double a);

void ddot_np_matrix(NP_MATRIX **c, NP_MATRIX *a, NP_MATRIX *b);

void test_print_np_matrix(NP_MATRIX *input);

#endif
