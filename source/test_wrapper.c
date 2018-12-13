#define DEBUG 1
#include "shared_types.h"
#include "shared_library.h" 
#include "blas_helper.h"
#include "omp_helper.h"

void test_print_arr(const RESTRICT doubleptr arr, const int m, const int n){
    int i;
    int j;
    for (i = 0; i < 3; ++i){
        for(j = 0; j < 3; ++j){
            fprintf(stderr, "%.3lf ", arr[i * n + j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "---------------------------\n");
}


#define N 4
void test_strided_copy_np_matrix(){
    NP_MATRIX a,b, bs,c, d;
    double *ra = malloc(sizeof(double) * N * N);
    double *rb = malloc(sizeof(double) * N * N * 4);
    double *rc = malloc(sizeof(double) * N * N);
    double *rd = malloc(sizeof(double) * N * N);

    int i;
    for (i = 0; i < N * N; ++i){
        ra[i] = i;
    }
    for (i = 0; i < N * N * 4; ++i){
        rb[i] = i + N * N;
    }

    wrap_np_matrix(&a, ra, N, N, N, 1);
    wrap_np_matrix(&b, rb, 2 * N, 2 * N, 2 * N, 1);
    wrap_np_matrix(&bs, rb, 2 * N, N, 2 * N, 2);
/*    wrap_np_matrix(&c, rc, N, N, N, 1);*/
    
    
    test_print_np_matrix(&a);
    test_print_np_matrix(&b);
    
    
    copy_np_matrix(&b, 3, 3, &a);
    
    test_print_np_matrix(&b);
/*    test_print_np_matrix(&c);*/
/*    */
/*    copy_np_matrix(&c, 0, 0, &a);*/
/*   */
/*    test_print_np_matrix(&c);   */
    
    free(ra);
    free(rb);
    free(rc);
    free(rd);
}

void test_index_np_matrix(){
    int i;
    NP_MATRIX *a = new_np_matrix(N, N);
    for (i = 0; i < N * N; ++i){
        a->matrix[i] = i + N * N;
    }
    test_print_np_matrix(a);
    
    fprintf(stdout, "a[%d][%d] = %lf\n", 2, 3, index_np_matrix(a, 2, 3));
    transpose_np_matrix(a);
    
    test_print_np_matrix(a);
    
    fprintf(stdout, "a[%d][%d] = %lf\n", 2, 3, index_np_matrix(a, 2, 3));

        
    delete_np_matrix(&a);
}

void test_axpy_np_matrix(){
    int i;
    NP_MATRIX *a = new_np_matrix(N, N * 2);
    NP_MATRIX *b = new_np_matrix(N * 2, N);
    NP_MATRIX *e = new_np_matrix(N * 2, N);
    NP_MATRIX *c = new_np_matrix(N, N * 2);
    NP_MATRIX *d = new_np_matrix(2 * N, N);
    for (i = 0; i < N * N * 2; ++i){
        a->matrix[i] = i + 10;
        b->matrix[i] = i + 100;
        e->matrix[i] = i + 50;
        
    }

    
    double alpha = 2.0;
    transpose_np_matrix(a);
    test_print_np_matrix(a);
    test_print_np_matrix(b);
    axpy_np_matrix(&a, b, alpha, a);
    test_print_np_matrix(a);
    

    delete_np_matrix(&a);
    delete_np_matrix(&b);
    delete_np_matrix(&c);    
    delete_np_matrix(&d); 
    delete_np_matrix(&e); 
}

void test_scal_np_matrix(){
    int i;
    NP_MATRIX *a = new_np_matrix(N, N * 2);
    NP_MATRIX *c = new_np_matrix(N, N * 2);
    NP_MATRIX *d = new_np_matrix(2 * N, N);
    for (i = 0; i < N * N * 2; ++i){
        a->matrix[i] = i + 10;
    }
    test_print_np_matrix(a);
    
    double alpha = 2.0;
    
    scal_np_matrix(&a, a, alpha);
    test_print_np_matrix(a);
    
    transpose_np_matrix(a);
    test_print_np_matrix(a);
    scal_np_matrix(&d, a, alpha);
    test_print_np_matrix(d);
 
    
    delete_np_matrix(&a);
    delete_np_matrix(&c);    
    delete_np_matrix(&d);    
}

void test_ddot_np_matrix(){
    int i;
    NP_MATRIX au, bu, ac, bc;
    NP_MATRIX *a = new_np_matrix(4, 3);
    NP_MATRIX *b = new_np_matrix(4, 2);
    NP_MATRIX *c = new_np_matrix(400, 200);
    double *ra = malloc(sizeof(double) * 240000);
    double *rb = malloc(sizeof(double) * 240000);
    for (i = 0; i < a->m * a->n; ++i){
        a->matrix[i] = i + 1;
        
    }
    for (i = 0; i < 240000; ++i){
        ra[i] = 1;
        rb[i] = 1;
    }
    
    
    
    for (i = 0; i < b->m * b->n; ++i){
        b->matrix[i] = i + 3;
    }    
    
    
    //transpose_np_matrix(b);
    //transpose_np_matrix(a);
    
    wrap_np_matrix(&au, ra, 400, 300, 600, 2);
    wrap_np_matrix(&bu, rb, 300, 200, 400, 2);
    wrap_np_matrix(&ac, ra, 400, 300, 600, 1);
    wrap_np_matrix(&bc, rb, 300, 200, 400, 1);
    //test_print_np_matrix(a);
    //test_print_np_matrix(&au);
    //test_print_np_matrix(&bu);
    //transpose_np_matrix(&as);
    
    double stime = omp_get_wtime();
    
    for (i = 0; i < 30; ++i){
        ddot_np_matrix(&c, &au, &bu);
    }

    flogf(stdout, GRN "U @ U time:" RESET " %lf\n", (omp_get_wtime() - stime));
    stime = omp_get_wtime();
    
    for (i = 0; i < 30; ++i){
        ddot_np_matrix(&c, &au, &bc);
    }

    flogf(stdout, GRN "U @ C time:" RESET " %lf\n", (omp_get_wtime() - stime));
    stime = omp_get_wtime();
    
    for (i = 0; i < 30; ++i){
        ddot_np_matrix(&c, &ac, &bu);
    }

    flogf(stdout, GRN "C @ U time:" RESET " %lf\n", (omp_get_wtime() - stime));
    stime = omp_get_wtime();
    
    for (i = 0; i < 30; ++i){
        ddot_np_matrix(&c, &ac, &bc);
    }

    flogf(stdout, GRN "C @ C time:" RESET " %lf\n", (omp_get_wtime() - stime));
    
    //test_print_np_matrix(c);
    
    //transpose_np_matrix(a);
    
    delete_np_matrix(&a);
    delete_np_matrix(&b);    
    delete_np_matrix(&c);  
}

void test_delete_np_matrix(NP_MATRIX **a){

}


void test_flip_ij(){
    NP_MATRIX *a = new_np_matrix(7,7);
    NP_MATRIX *b = NULL;
    int i;
    int bd[2] = {3,4};
    for (i = 0; i < 49; ++i){
        a->matrix[i] = i;
    }
    transpose_np_matrix(a);
    test_print_np_matrix(a);
    flip_ij(&b, 0, 1, a, bd);
    test_print_np_matrix(b);
    
}

int main(){
    test_ddot_np_matrix();
}

