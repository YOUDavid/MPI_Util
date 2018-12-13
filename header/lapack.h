#ifndef LAPACK_H
#define LAPACK_H
#include <shared_types.h>
extern void daxpy_(constintconstptr N, constdoubleconstptr DA, constdoubleconstptr DX, constintconstptr INCX, doubleconstptr DY, constintconstptr INCY);

extern void dscal_(constintconstptr	N, constdoubleconstptr DA, doubleconstptr DX, constintconstptr INCX);

extern void dcopy_(constintconstptr N, constdoubleconstptr DX, constintconstptr INCX, doubleconstptr DY, constintconstptr INCY);

extern void dgemm_(constcharconstptr TRANSA, constcharconstptr TRANSB, constintconstptr M, constintconstptr N, constintconstptr K, constdoubleconstptr ALPHA, constdoubleconstptr A, constintconstptr LDA, constdoubleconstptr B, constintconstptr LDB, constdoubleconstptr BETA, doubleconstptr C, constintconstptr LDC); 

extern void dgemv_(constcharconstptr TRANSA, constintconstptr M, constintconstptr N, constdoubleconstptr ALPHA, constdoubleconstptr A, constintconstptr LDA, constdoubleconstptr DX, constintconstptr INCX, constdoubleconstptr BETA, doubleconstptr DY, constintconstptr INCY); 

extern double ddot_(constintconstptr N, constdoubleconstptr DX, constintconstptr INCX, doubleconstptr DY, constintconstptr INCY);

/*extern void daxpy_(int* N, double* DA, double* DX, int* INCX, double* DY, int* INCY);*/

/*extern void dscal_(int*	N, double* DA, double* DX, int* INCX);*/

/*extern void dcopy_(int* N, double* DX, int* INCX, double* DY, int* INCY);*/

/*extern void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC); */

#endif
