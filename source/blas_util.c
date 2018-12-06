#include "shared_types.h"
#include "shared_library.h" 
#include "blas_helper.h"
#include "omp_helper.h"
/*#include <mkl.h>*/
///Actual matrix objects on heap
///View of matrix objects on stack
void test_print_np_matrix(NP_MATRIX *input){
    int i;
    int j;
    int cont = 0;
    double result = 0;
    fprintf(stderr, "matrix: m=%d, n=%d, tra=%c\n", input->m, input->n, input->tra);
    if (input->tra == 'N'){
        for (i = 0; i < input->m; ++i){
            for(j = 0; j < input->n; ++j){
            
                result += input->matrix[i * input->n + j];
                //fprintf(stderr, "%4.1lf ", input->matrix[i * input->n + j]);
                //cont++;
                //if (cont % 4 == 0){
                //    fprintf(stderr, "\n  ");
                //    cont = 0;
                //}
/*                if (input->matrix[i * input->n + j] > 0){*/
/*                    fprintf(stderr, "O");*/
/*                } else{*/
/*                    fprintf(stderr, "X");*/
/*                }*/
            }
            //fprintf(stderr, "\n");
        }
    } else{
        for (i = 0; i < input->m; ++i){
            for(j = 0; j < input->n; ++j){
                 result += input->matrix[i + input->m * j];
/*                fprintf(stderr, "%15.8le ", input->matrix[i + input->m * j]);*/
/*                cont++;*/
/*                if (cont % 4 == 0){*/
/*                    fprintf(stderr, "\n  ");*/
/*                    cont = 0;*/
/*                }*/
/*                if (input->matrix[i * input->n + j] > 0){*/
/*                    fprintf(stderr, "O");*/
/*                } else{*/
/*                    fprintf(stderr, "X");*/
/*                }*/
            }
            //fprintf(stderr, "\n");
        }    
    }
    fprintf(stderr, "sum:%lf\n", result);
    fprintf(stderr, "---------------------------\n");
}


void generate_supermatrix(NP_MATRIX **ptr2old, int i, int j, int k, int l, 
                          double **matrices, int *blockdim, int ndim, char *tras){
    NP_MATRIX ul, ur, ll, lr;
    NP_MATRIX *new = renew_np_matrix(ptr2old, blockdim[i] + blockdim[j], blockdim[k] + blockdim[l]);
    wrap_np_matrix(&ul, matrices[i * ndim + k], blockdim[i], blockdim[k], tras[i * ndim + k]);
    wrap_np_matrix(&ll, matrices[j * ndim + k], blockdim[j], blockdim[k], tras[j * ndim + k]);
    wrap_np_matrix(&ur, matrices[i * ndim + l], blockdim[i], blockdim[l], tras[i * ndim + l]);
    wrap_np_matrix(&lr, matrices[j * ndim + l], blockdim[j], blockdim[l], tras[j * ndim + l]);
    copy_np_matrix(new, 0, 0, &ul);
    copy_np_matrix(new, 0, ul.n, &ur);
    copy_np_matrix(new, ul.m, 0, &ll);
    copy_np_matrix(new, ul.m, ul.n, &lr);
}

void flip_ij(NP_MATRIX **ptr2old, const int i, const int j, NP_MATRIX *matrix, int *blockdim){
    NP_MATRIX *new = renew_np_matrix(ptr2old, blockdim[i] + blockdim[j], blockdim[i] + blockdim[j]);
    NP_MATRIX t_matrix;
    wrap_np_matrix(&t_matrix, matrix->matrix, matrix->m, matrix->n, matrix->tra);
    transpose_np_matrix(&t_matrix);
    strided_copy_np_matrix(new, blockdim[j], blockdim[j], &t_matrix, blockdim[i], blockdim[i], 0, 0);
    strided_copy_np_matrix(new, 0, 0, &t_matrix, blockdim[j], blockdim[j], blockdim[i], blockdim[i]);
    strided_copy_np_matrix(new, 0, blockdim[j], &t_matrix, blockdim[j], blockdim[i], blockdim[i], 0);
    strided_copy_np_matrix(new, blockdim[j], 0, &t_matrix, blockdim[i], blockdim[j], 0, blockdim[i]);
}

void transpose_np_matrix(NP_MATRIX *old){
    int temp = old->m;
    old->m = old->n;
    old->n = temp;
    if (old->tra == 'N'){
        old->tra = 'T';
    } else if (old->tra == 'T'){
        old->tra = 'N';
    } else {
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, old->tra);
    }
}

void copy_np_matrix(NP_MATRIX *des, const int row, const int col, NP_MATRIX *src){
    strided_copy_np_matrix(des, row, col, src, src->m, src->n, 0, 0);
}

void strided_copy_np_matrix(NP_MATRIX *des, const int drow, const int dcol, NP_MATRIX *src, const int realm, const int realn, const int srow, const int scol){
    //realm and realn denote the real size of the submatrix in src
    int i, des_ost, src_ost;
    int inc = 1;
    
    
    if (des->tra == 'N'){
        des_ost = drow * des->n + dcol; 
        if(src->tra == 'N'){
            src_ost = srow * src->n + scol; 
            #pragma omp parallel for
            for (i = 0; i < realm; ++i){
                dcopy_(&realn, src->matrix + src_ost + src->n * i, &inc, des->matrix + des_ost + des->n * i, &inc);
            }
        } else if (src->tra == 'T'){
            src_ost = srow + src->m * scol; 
            #pragma omp parallel for
            for (i = 0; i < realm; ++i){
                dcopy_(&realn, src->matrix + src_ost + i, &src->m, des->matrix + des_ost + des->n * i, &inc);
            }
        } else {
            fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, src->tra);
        }
    
    } else if (des->tra == 'T'){
        des_ost = drow + des->m * dcol; 
        if(src->tra == 'N'){
            src_ost = srow * src->n + scol; 
            #pragma omp parallel for
            for (i = 0; i < realm; ++i){
                dcopy_(&realn, src->matrix + src_ost + src->n * i, &inc, des->matrix + des_ost + i, &des->m);
            }
        } else if (src->tra == 'T'){
            src_ost = srow + src->m * scol; 
            #pragma omp parallel for
            for (i = 0; i < realn; ++i){
                dcopy_(&realm, src->matrix + src_ost + src->m * i, &inc, des->matrix + des_ost + des->m * i, &inc);
            }
        } else {
            fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, src->tra);
        }
    } else{
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, des->tra);
    }
} 

void delete_np_matrix(NP_MATRIX **ptr2old){
    //delete the NP_MATRIX stored by the pointer, take the address of pointer
    //as input
    if (*ptr2old != NULL){
        if ((*ptr2old)->matrix != NULL){
            free((*ptr2old)->matrix);
            (*ptr2old)->matrix = NULL;
        } else{
            fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "free null matrix!");    
        }
        free(*ptr2old);
        *ptr2old = NULL;
    } else{
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "deleting null matrix!");
    }
}

NP_MATRIX *new_np_matrix(const int m, const int n){
    //return a new pointer pointing to a new NP_MATRIX of size m X n
    //fprintf(stderr, "%s: %s: want to alloc matrix size of %d by %d\n", __FILE__, __func__, m, n);
    NP_MATRIX *result = malloc(sizeof(NP_MATRIX));
    if (result == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for NP_MATRIX failed!");
    } 
    result->matrix = malloc(sizeof(double) * m * n);
    if (result->matrix == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for matrix failed!");
    }   
    result->m = m;
    result->n = n;
    result->tra = 'N';
    return result;
    
}

NP_MATRIX *cnew_np_matrix(const int m, const int n){
    //return a new pointer pointing to a new NP_MATRIX of size m X n, zero-init
    NP_MATRIX *result = malloc(sizeof(NP_MATRIX));
    if (result == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for NP_MATRIX failed!");
    }
    result->matrix = calloc(m * n, sizeof(double));
    if (result->matrix == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "calloc for matrix failed!");
    }        
    
    result->m = m;
    result->n = n;
    result->tra = 'N';
    return result;
}

NP_MATRIX *renew_np_matrix(NP_MATRIX **ptr2old, const int m, const int n){
    //renew the NP_MATRIX stored by the pointer, take the address of pointer
    //as input
    //fprintf(stderr, "%s: %s: want to realloc matrix size of %d by %d\n", __FILE__, __func__, m, n);
    if (*ptr2old == NULL){
        *ptr2old = new_np_matrix(m, n);
    } else if(m * n > (*ptr2old)->m * (*ptr2old)->n){
        (*ptr2old)->matrix = realloc((*ptr2old)->matrix, sizeof(double) * m * n);
        if ((*ptr2old)->matrix == NULL){
            fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "realloc for matrix failed!");
        }  
        (*ptr2old)->m = m;
        (*ptr2old)->n = n;
        (*ptr2old)->tra = 'N';
    } else{
        (*ptr2old)->m = m;
        (*ptr2old)->n = n;
        (*ptr2old)->tra = 'N';
    }
    return *ptr2old;
}

void wrap_np_matrix(NP_MATRIX *wrapper, double *raw, const int m, const int n, const char tra){
    wrapper->matrix = raw;
    wrapper->m = m;
    wrapper->n = n;
    wrapper->tra = tra;
}

double index_np_matrix(NP_MATRIX *matrix, const int i, const int j){
/*    if (i >= matrix->m){*/
/*        fprintf(stderr, "%s: dimension error: matrix->m=%d, i=%d\n", __func__, matrix->m, i);*/
/*    }*/
/*    */
/*    if (j >= matrix->n){*/
/*        fprintf(stderr, "%s: dimension error: matrix->n=%d, j=%d\n", __func__, matrix->n, j);*/
/*    }    */
    
    if (matrix->tra == 'N'){
        return matrix->matrix[i * matrix->n + j];
    } else if (matrix->tra == 'T'){
        return matrix->matrix[i + matrix->m * j];
    } else{
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, matrix->tra);
    }
    return 0.0;
}

void axpy_np_matrix(NP_MATRIX **ptr2result, NP_MATRIX *x, const double a, NP_MATRIX *y){
    int i;
    int inc = 1;
    NP_MATRIX *result = *ptr2result;
/*    if (x->m != y->m){*/
/*        fprintf(stderr, "%s: dimension error: x->m=%d, y->m=%d\n", __func__, x->m, y->m);*/
/*    }*/
/*    */
/*    if (x->n != y->n){*/
/*        fprintf(stderr, "%s: dimension error: x->n=%d, y->n=%d\n", __func__, x->n, y->n);*/
/*    }*/
    
    if (result != y && result != x){
        result = renew_np_matrix(ptr2result, x->m, x->n);
        if (x->tra == 'N' && y->tra == 'N'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                dcopy_(&y->n, y->matrix + i * y->n, &inc, result->matrix + i * result->n, &inc);
                daxpy_(&x->n, &a, x->matrix + i * x->n, &inc, result->matrix + i * result->n, &inc);
            }
        } else if (x->tra == 'T' && y->tra == 'T'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                dcopy_(&y->n, y->matrix + i, &y->m, result->matrix + i * result->n, &inc);
                daxpy_(&x->n, &a, x->matrix + i, &x->m, result->matrix + i * result->n, &inc);
            }    
        } else if (x->tra == 'T' && y->tra == 'N'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                dcopy_(&y->n, y->matrix + i * y->n, &inc, result->matrix + i * result->n, &inc);
                daxpy_(&x->n, &a, x->matrix + i, &x->m, result->matrix + i * result->n, &inc);
            }  
        } else if (x->tra == 'N' && y->tra == 'T'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                dcopy_(&y->n, y->matrix + i, &y->m, result->matrix + i * result->n, &inc);
                daxpy_(&x->n, &a, x->matrix + i * x->n, &inc, result->matrix + i * result->n, &inc);
            }      
        } else {
            fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c and %c\n", __func__, x->tra, y->tra);
        }
    } else if (result == y){
        //inplace operation
        if (x->tra == 'N' && y->tra == 'N'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                daxpy_(&x->n, &a, x->matrix + i * x->n, &inc, y->matrix + i * y->n, &inc);
            }
        } else if (x->tra == 'T' && y->tra == 'T'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                daxpy_(&x->n, &a, x->matrix + i, &x->m, y->matrix + i, &y->m);
            }    
        } else if (x->tra == 'T' && y->tra == 'N'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                daxpy_(&x->n, &a, x->matrix + i, &x->m, y->matrix + i * y->n, &inc);
            }  
        } else if (x->tra == 'N' && y->tra == 'T'){
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                daxpy_(&x->n, &a, x->matrix + i * x->n, &inc, y->matrix + i, &y->m);
            }      
        } else {
            fprintf(stderr, "%s: inplace data structure error: unrecognized tra tag: %c and %c\n", __func__, x->tra, y->tra);
        }        
        
    }
}

void scal_np_matrix(NP_MATRIX **ptr2result, NP_MATRIX *x, const double a){
    NP_MATRIX *result = *ptr2result;
    int i;
    int inc = 1;
    if (x->tra == 'N'){
        result = renew_np_matrix(ptr2result, x->m, x->n);
        #pragma omp parallel for
        for (i = 0; i < x->m; ++i){
            dcopy_(&x->n, x->matrix + i * x->n, &inc, result->matrix + i * result->n, &inc);
            dscal_(&result->n, &a, result->matrix + i * result->n, &inc);
        }
    } else if (x->tra == 'T'){
        result = renew_np_matrix(ptr2result, x->m, x->n);
        #pragma omp parallel for
        for (i = 0; i < x->m; ++i){
            dcopy_(&x->n, x->matrix + i, &x->m, result->matrix + i * result->n, &inc);
            dscal_(&result->n, &a, result->matrix + i * result->n, &inc);
        }
    } else{
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, x->tra);
    }
}

void ddot_np_matrix(NP_MATRIX **ptr2c, NP_MATRIX *a, NP_MATRIX *b){
    int lda, ldb, ldc, offseta, offsetb, offsetc;
    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    if (a->tra == 'N'){
        lda = a->n;
    } else if (a->tra == 'T'){
        lda = a->m;
    } else{
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, a->tra);
    }

    if (b->tra == 'N'){
        ldb = b->n;
    } else if (b->tra == 'T'){
        ldb = b->m;
    } else{
        fprintf(stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, b->tra);
    }   
    
/*    if (a->n != b-> m){*/
/*        fprintf(stderr, "%s: dimension error: a->n=%d, b->m=%d\n", __func__, a->n, b->m);*/
/*    }*/
    
    NP_MATRIX *c; 
    c = renew_np_matrix(ptr2c, a->m, b->n);
    ldc = c->n;
    
    offseta = 0;
    offsetb = 0;
    offsetc = 0;
    NPdgemm(b->tra, a->tra, b->n, a->m, a->n, ldb, lda, ldc, offsetb, offseta, offsetc, b->matrix, a->matrix, c->matrix, alpha, beta);
}
