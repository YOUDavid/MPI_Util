#ifndef MPI_HELPER_H
#define MPI_HELPER_H

#include <mpi.h>
#include "shared_types.h"
#include "shared_library.h"

void mpi_lNPdgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num, int * completed);
                  
void mpi_lOMPdgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num, int * completed);
                  
void mpi_ldgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num, int * completed);                  
                                    
void mpi_print(int len, double* arr);                  
                  
void mpi_setONT(int ont);

void mpi_init();

void mpi_final();


#endif
