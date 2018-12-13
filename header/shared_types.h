#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))
#define TRUE 1
#define FALSE 0
#define SIGNAL_LNPDGEMM 1
#define SIGNAL_LPRESIDUE 2
#define SIGNAL_LOMPDGEMM 3
#define SIGNAL_SETONT 4
#define SIGNAL_PRINT 0

#define META_TAG 0
#define DATA_A_TAG 1
#define DATA_B_TAG 2
#define RESULT_TAG 3
#define SIGNAL_TAG 4
#define NUM_OF_TAGS 5
#define NUM_OF_CTHREADS 1
#define NUM_OF_WTHREADS 1
#define CGRAPE_SIZE 1
#define WGRAPE_SIZE 1

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

#define CHUNCK_SIZE 1500 * 1500

#define RESTRICT __restrict

typedef __attribute__((aligned(32))) int* intptr;
typedef __attribute__((aligned(32))) const int *const constintconstptr;
typedef __attribute__((aligned(32))) int *const intconstptr;
typedef __attribute__((aligned(32))) const int* constintptr;

typedef __attribute__((aligned(64))) double* doubleptr;
typedef __attribute__((aligned(64))) const double *const constdoubleconstptr;
typedef __attribute__((aligned(64))) double *const doubleconstptr;
typedef __attribute__((aligned(64))) const double* constdoubleptr;

typedef __attribute__((aligned(64))) doubleptr* doubleptrptr;
typedef __attribute__((aligned(64))) const doubleptr *const constdoubleptrconstptr;
typedef __attribute__((aligned(64))) doubleptr *const doubleptrconstptr;
typedef __attribute__((aligned(64))) const doubleptr* constdoubleptrptr;

typedef __attribute__((aligned(8))) char* charptr;
typedef __attribute__((aligned(8))) const char *const constcharconstptr;
typedef __attribute__((aligned(8))) char *const charconstptr;
typedef __attribute__((aligned(8))) const char* constcharptr;

/*extern void dgemm_(const char *, const char *,*/
/*            const int *, const int *, const int *,*/
/*            const double *, const double *, const int *,*/
/*            const double *, const int *,*/
/*            const double *, double *, const int *);*/

#define NBLOCKS_MATRIX_META 13
struct _MATRIX_META {
	char tra_a, tra_b;
	int m, n, k, lda, ldb, ldc, offseta, offsetb, offsetc;
	double alpha, beta;
};
typedef struct _MATRIX_META MATRIX_META;

struct _NP_MATRIX {
    char tra; //DEPRECATED
    int m, n, mstride, nstride;
    double *matrix;
};
typedef struct _NP_MATRIX NP_MATRIX;

#if DEBUG > 0
#define flogf(outsource, format, ...) fprintf(outsource, "%s: %s: "format, __FILE__, __func__, ##__VA_ARGS__)

#define logf(format, ...) printf("%s: %s: "format, __FILE__, __func__, ##__VA_ARGS__)

#define elogf(format, ...) fprintf(stderr, "%s: %s: " format, __FILE__, __func__, ##__VA_ARGS__)
 
#define condflogf(condition, outsource, format, ...) if (condition) fprintf(outsource, "%s: %s: "format, __FILE__, __func__, ##__VA_ARGS__)

#else
#define flogf(outsource, format, ...)  

#define logf(format, ...)  

#define elogf(format, ...)  

#define condflogf(condition, outsource, format, ...) 
 
#endif


#endif
