#define DEBUG 0
#include "shared_types.h"
#include "shared_library.h" 
#include "blas_helper.h"
#include "omp_helper.h"

///Actual matrix objects on heap
///View of matrix objects on stack
void test_print_np_matrix(NP_MATRIX *input){
    //stride added
    int i;
    int j;
    fprintf(stderr,BLU "matrix: m=%d, n=%d\n" RESET, input->m, input->n);
    for (i = 0; i < input->m; ++i){
        for(j = 0; j < input->n; ++j){
            fprintf(stderr,RED "%4.1lf " RESET, input->matrix[i * input->mstride + j * input->nstride]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr,GRN "---------------------------\n" RESET);
}


void generate_supermatrix(NP_MATRIX **ptr2old, const int i, const int j, const int k, const int l, constdoubleptrconstptr matrices, constintconstptr blockdim, const int ndim, const int mat_len){
    NP_MATRIX ul, ur, ll, lr;
    NP_MATRIX *new = renew_np_matrix(ptr2old, blockdim[i] + blockdim[j], blockdim[k] + blockdim[l]);
    wrap_np_matrix(&ul, matrices[i * ndim + k], blockdim[i], blockdim[k], 1, 1);
    wrap_np_matrix(&ll, matrices[j * ndim + k], blockdim[j], blockdim[k], 1, 1);
    wrap_np_matrix(&ur, matrices[i * ndim + l], blockdim[i], blockdim[l], 1, 1);
    wrap_np_matrix(&lr, matrices[j * ndim + l], blockdim[j], blockdim[l], 1, 1);
    copy_np_matrix(new, 0, 0, &ul);
    copy_np_matrix(new, 0, ul.n, &ur);
    copy_np_matrix(new, ul.m, 0, &ll);
    copy_np_matrix(new, ul.m, ul.n, &lr);
}

void flip_ij(NP_MATRIX **ptr2old, const int i, const int j, NP_MATRIX *matrix, constintconstptr blockdim){
    NP_MATRIX *new = renew_np_matrix(ptr2old, blockdim[i] + blockdim[j], blockdim[i] + blockdim[j]);
    NP_MATRIX t_matrix;
    wrap_np_matrix(&t_matrix, matrix->matrix, matrix->m, matrix->n, 1, 1);
    transpose_np_matrix(&t_matrix);
    strided_copy_np_matrix(new, blockdim[j], blockdim[j], &t_matrix, blockdim[i], blockdim[i], 0, 0);
    strided_copy_np_matrix(new, 0, 0, &t_matrix, blockdim[j], blockdim[j], blockdim[i], blockdim[i]);
    strided_copy_np_matrix(new, 0, blockdim[j], &t_matrix, blockdim[j], blockdim[i], blockdim[i], 0);
    strided_copy_np_matrix(new, blockdim[j], 0, &t_matrix, blockdim[i], blockdim[j], 0, blockdim[i]);
}

void transpose_np_matrix(NP_MATRIX *old){
    //Stride added
    int temp;
    
    temp = old->m;
    old->m = old->n;
    old->n = temp;
    
    temp = old->mstride;
    old->mstride = old->nstride;
    old->nstride = temp;
}

void copy_np_matrix(NP_MATRIX *des, const int row, const int col, NP_MATRIX *src){
    strided_copy_np_matrix(des, row, col, src, src->m, src->n, 0, 0);
}

void strided_copy_np_matrix(NP_MATRIX *des, const int drow, const int dcol, NP_MATRIX *src, const int realm, const int realn, const int srow, const int scol){
    //realm and realn denote the real size of the submatrix in src
    //stride added
    int i;
    double *des_ost, *src_ost;
    
    
    des_ost = des->matrix + drow * des->mstride + dcol * des->nstride;
    src_ost = src->matrix + srow * src->mstride + scol * src->nstride;
    
    //potential improvement on which dimension to parallelize
    #pragma omp parallel for
    for (i = 0; i < realm; ++i){
        dcopy_(&realn, src_ost + src->mstride * i, &src->nstride, des_ost + des->mstride * i, &des->nstride);
    }
} 

void delete_np_matrix(NP_MATRIX **ptr2old){
    //delete the NP_MATRIX stored by the pointer, take the address of pointer as input
    //Stride added
    
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
    //Stride added
    
    flogf(stderr, "%s: %s: want to alloc matrix size of %d by %d\n", __FILE__, __func__, m, n);
    
    NP_MATRIX *result = malloc(sizeof(NP_MATRIX));
    if (result == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for NP_MATRIX failed!");
    } 
    result->matrix = malloc(sizeof(double) * m * n);
    if (result->matrix == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for matrix failed!");
    }
    result->mstride = n;
    result->nstride = 1; 
    result->m = m;
    result->n = n;
    result->tra = 'N';
    return result;
}

NP_MATRIX *cnew_np_matrix(const int m, const int n){
    //return a new pointer pointing to a new NP_MATRIX of size m X n, zero-init
    //Stride added
    
    NP_MATRIX *result = malloc(sizeof(NP_MATRIX));
    if (result == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "malloc for NP_MATRIX failed!");
    }
    result->matrix = calloc(m * n, sizeof(double));
    if (result->matrix == NULL){
        fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "calloc for matrix failed!");
    }        
    result->mstride = n;
    result->nstride = 1; 
    result->m = m;
    result->n = n;
    result->tra = 'N';
    return result;
}

NP_MATRIX *renew_np_matrix(NP_MATRIX **ptr2old, const int m, const int n){
    //renew the NP_MATRIX stored by the pointer, take the address of pointer as input
    //Stride added
    
    flogf(stderr, "%s: %s: want to realloc matrix size of %d by %d\n", __FILE__, __func__, m, n);
    if (*ptr2old == NULL){
        *ptr2old = new_np_matrix(m, n);
    } else if(m * n > (*ptr2old)->m * (*ptr2old)->n){
        NP_MATRIX *new = realloc((*ptr2old)->matrix, sizeof(double) * m * n);
        
        if ((*ptr2old)->matrix == NULL){
            fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "realloc for matrix failed!");
        } else{
            *ptr2old = new;
        }  
        (*ptr2old)->m = m;
        (*ptr2old)->n = n;
        (*ptr2old)->mstride = n;
        (*ptr2old)->nstride = 1;
        (*ptr2old)->tra = 'N';
    } else{
        (*ptr2old)->m = m;
        (*ptr2old)->n = n;
        (*ptr2old)->mstride = n;
        (*ptr2old)->nstride = 1;        
        (*ptr2old)->tra = 'N';
    }
    return *ptr2old;
}

void wrap_np_matrix(NP_MATRIX *wrapper, doubleconstptr raw, const int m, const int n, const int mstride, const int nstride){
    //stride added
    wrapper->matrix = raw;
    wrapper->m = m;
    wrapper->n = n;
    wrapper->mstride = mstride;
    wrapper->nstride = nstride;
}

double index_np_matrix(NP_MATRIX *matrix, const int i, const int j){
    //Stride added

    condflogf(i >= matrix->m, stderr, "%s: dimension error: matrix->m=%d, i=%d\n", __func__, matrix->m, i);
    condflogf(j >= matrix->n, stderr, "%s: dimension error: matrix->n=%d, j=%d\n", __func__, matrix->n, j);
    //condflogf(matrix->tra != 'N' || matrix->tra != 'T', stderr, "%s: data structure error: unrecognized tra tag: %c\n", __func__, matrix->tra);
    
    return matrix->matrix[i * matrix->mstride + j * matrix->nstride];
}

void axpy_np_matrix(NP_MATRIX **ptr2result, NP_MATRIX *x, const double a, NP_MATRIX *y){
    //Stride added
    int i;
    NP_MATRIX *result = *ptr2result;
    
    condflogf(x->m != y->m, stderr, "%s: dimension error: x->m=%d, y->m=%d\n", __func__, x->m, y->m);
    condflogf(x->n != y->n, stderr, "%s: dimension error: x->n=%d, y->n=%d\n", __func__, x->n, y->n);
    //condflogf(x->tra != 'N' || x->tra != 'T' || y->tra != 'N' || y->tra != 'T', stderr, "%s: data structure error: unrecognized tra tag: %c and %c\n", __func__, x->tra, y->tra);
    
    if (result != y){
        if(result != x){
            //out-of-place operation, z = a * x + y
            result = renew_np_matrix(ptr2result, x->m, x->n);
            #pragma omp parallel for
            for (i = 0; i < x->m; ++i){
                dcopy_(&x->n, y->matrix + i * y->mstride, &y->nstride, result->matrix + i * result->mstride, &result->nstride);
                daxpy_(&x->n, &a, x->matrix + i * x->mstride, &x->nstride, result->matrix + i * result->mstride, &result->nstride);
            }
        } else{
            //in-place operation, x = a * x + y
            scal_np_matrix(ptr2result, x, a);
            axpy_np_matrix(ptr2result, y, 1.0, x);
        }
    } else if (result != x){
        //in-place operation, y = a * x + y
        #pragma omp parallel for
        for (i = 0; i < x->m; ++i){
            daxpy_(&x->n, &a, x->matrix + i * x->mstride, &x->nstride, y->matrix + i * y->mstride, &y->nstride);
        }
    } else{
        //Why don't you call scal_np_matrix???  x = a * x + x
        scal_np_matrix(ptr2result, x, a + 1.0);
    }
}

void scal_np_matrix(NP_MATRIX **ptr2result, NP_MATRIX *x, const double a){
    //stride added
    NP_MATRIX *result = *ptr2result;
    int i;
    int inc = 1;
    if (result != x){
        //out-of-place operation, y = a * x
        result = renew_np_matrix(ptr2result, x->m, x->n);
        #pragma omp parallel for
        for (i = 0; i < x->m; ++i){
            dcopy_(&x->n, x->matrix + i * x->mstride, &x->nstride, result->matrix + i * result->mstride, &result->nstride);
            dscal_(&result->n, &a, result->matrix + i * result->mstride, &result->nstride);
        }        
    } else{
        //in-place operation, x = a * x
        #pragma omp parallel for
        for (i = 0; i < x->m; ++i){
            dscal_(&result->n, &a, result->matrix + i * result->mstride, &result->nstride);
        }  
    }
}

void ddot_np_matrix(NP_MATRIX **ptr2c, NP_MATRIX *a, NP_MATRIX *b){
    //stride added
    
    condflogf(a->n != b->m, stderr, "%s: dimension error: a->n=%d, b->m=%d\n", __func__, a->n, b->m);
    char tra_a = 'U', tra_b = 'U';
    int lda, ldb, ldc, i, j;
    double alpha = 1.0, beta = 0.0;
    NP_MATRIX *temp_a = NULL;
    NP_MATRIX *temp_b = NULL;
    
    if (a->mstride > a->nstride){
        //row major order
        if (a->nstride == 1){
            //fully contiguous
            tra_a = 'N';
            lda = a->mstride;
        } else{
            //not fully contiguous            
        }
    } else if (a->mstride < a->nstride){
        //column major order
        if (a->mstride == 1){
            //fully contiguous
            tra_a = 'T';
            lda = a->nstride;
        } else{
            //not fully contiguous
        }        
    } else{
        fprintf(stderr, "%s: stride error: a->mstride=%d, a->nstride=%d\n", __func__, a->mstride, a->nstride);
    }
    
    
    
    if (b->mstride > b->nstride){
        //row major order
        if (b->nstride == 1){
            //fully contiguous
            tra_b = 'N';
            ldb = b->mstride;
        } else{
            //not fully contiguous
        }
    } else if (b->mstride < b->nstride){
        //column major order
        if (b->mstride == 1){
            //fully contiguous
            tra_b = 'T';
            ldb = b->nstride;
        } else{
            //not fully contiguous
        }        
    } else{
        fprintf(stderr, "%s: stride error: b->mstride=%d, b->nstride=%d\n", __func__, b->mstride, b->nstride);
    }

    
    if (tra_a != 'U'){
        if (tra_b != 'U'){
            NP_MATRIX *c = renew_np_matrix(ptr2c, a->m, b->n);
            ldc = c->mstride;
            NPdgemm(tra_b, tra_a, b->n, a->m, a->n, ldb, lda, ldc, 0, 0, 0, b->matrix, a->matrix, c->matrix, alpha, beta);
        } else{
            /*
            if (tra_a == 'N'){
                tra_a = 'T';
                lda = a->mstride;
            } else{
                tra_a = 'N';
                lda = a->nstride;
            }
            #pragma omp parallel for
            for (j = 0; j < c->n; ++j){
                dgemv_(&tra_a, &a->n, &a->m, &alpha, a->matrix, &lda, b->matrix + b->nstride * j, &b->mstride, &beta, c->matrix + c->nstride * j, &c->mstride);
            }
            */
            
            NP_MATRIX *bc = new_np_matrix(b->m, b->n);
            copy_np_matrix(bc, 0, 0, b);
            ddot_np_matrix(ptr2c, a, bc);
            delete_np_matrix(&bc);
            
        }
    } else{
        if (tra_b != 'U'){
            /*
            if (tra_b == 'N'){
                ldb = b->mstride;
            } else{
                ldb = b->nstride;
            }
            #pragma omp parallel for
            for (i = 0; i < c->m; ++i){
                dgemv_(&tra_b, &b->n, &b->m, &alpha, b->matrix, &ldb, a->matrix + a->mstride * i, &a->nstride, &beta, c->matrix + c->mstride * i, &c->nstride);
            }
            */
            
            NP_MATRIX *ac = new_np_matrix(a->m, a->n);
            copy_np_matrix(ac, 0, 0, a);
            ddot_np_matrix(ptr2c, ac, b);
            delete_np_matrix(&ac);
            
        } else{
            /*
            NP_MATRIX *c = renew_np_matrix(ptr2c, a->m, b->n);
            #pragma omp parallel for
            for (i = 0; i < c->m * c->n; ++i){
                c->matrix[(i % c->m) * c->mstride + (i / c->m) * c->nstride] = ddot_(&a->n, a->matrix + (i % c->m) * a->mstride, &a->nstride, b->matrix + (i / c->m) * b->nstride, &b->mstride);
            }
            */
            NP_MATRIX *ac = new_np_matrix(a->m, a->n);
            NP_MATRIX *bc = new_np_matrix(b->m, b->n);
            copy_np_matrix(ac, 0, 0, a);
            copy_np_matrix(bc, 0, 0, b);
            ddot_np_matrix(ptr2c, ac, bc);
            delete_np_matrix(&ac);
            delete_np_matrix(&bc);
            
        }
    }
}
