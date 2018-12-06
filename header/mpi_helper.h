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
                  
void mpi_lpresidue(int len_pl_tdi, 
                   int* completed, int* pairlist, 
                   double** T_di, int* m_tdi, int* n_tdi, char* tra_tdi, 
                   double** result, int* m_result, int* n_result, char* tra_result,
                   int len_s_f,
                   double** S_matrix, int *m_sm, int *n_sm, char* tra_sm,
                   double** F_matrix, int *m_fm, int *n_fm, char* tra_fm,
                   double* loc_fork, int m_lf, int n_lf, char tra_lf,
                   int len_tdim, int* t_dim,
                   int mfno, int nonredundant);             
                                    
void mpi_print(int len, double* arr);                  
                  
void mpi_setONT(int ont);

void mpi_init();

void mpi_final();


#endif
