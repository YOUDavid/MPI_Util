#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))
#define TRUE 1
#define FALSE 0
#define SIGNAL_LNPDGEMM 1
#define SIGNAL_LDGEMM 2
#define SIGNAL_LOMPDGEMM 3
#define SIGNAL_SETONT 4
#define SIGNAL_PRINT 0

#define META_TAG 0
#define DATA_A_TAG 1
#define DATA_B_TAG 2
#define RESULT_TAG 3
#define SIGNAL_TAG 4
#define NUM_OF_TAGS 5
#define NUM_OF_CTHREADS 16
#define NUM_OF_WTHREADS 0
#define CGRAPE_SIZE 4
#define WGRAPE_SIZE 16


#define CHUNCK_SIZE 1350 * 1350

#define RESTRICT __restrict

typedef __attribute__((aligned(32))) int* intptr;
typedef __attribute__((aligned(64))) double* doubleptr;
typedef __attribute__((aligned(64))) doubleptr* doubleptrptr;
typedef __attribute__((aligned(8))) char* charptr;

extern void dgemm_(const char *, const char *,
            const int *, const int *, const int *,
            const double *, const double *, const int *,
            const double *, const int *,
            const double *, double *, const int *);

#define NBLOCKS_MATRIX_META 13
struct _MATRIX_META {
	char tra_a, tra_b;
	int m, n, k, lda, ldb, ldc, offseta, offsetb, offsetc;
	double alpha, beta;
};
typedef struct _MATRIX_META MATRIX_META;


#endif
