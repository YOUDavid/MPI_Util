#include "shared_types.h"
#include "shared_library.h" 
#include "blas_helper.h"

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
    //TODO: TEST flip_ij and generate_supermatrix
    NP_MATRIX a,b,c,d,f,g;
    double *ra = malloc(sizeof(double) * N * N);
    double *rb = malloc(sizeof(double) * (N - 1) * (N - 1));
    double *rc = malloc(sizeof(double) * (N - 1) * N);
    double *rd = malloc(sizeof(double) * (N - 1) * N);
    double *rf = malloc(sizeof(double) * (N * N * 4 - 4 * N + 1));
    double *rg = malloc(sizeof(double) * (N * N * 4 - 4 * N + 1));
    int i;
    for (i = 0; i < N * N; ++i){
        ra[i] = i + N * N * 1;
    }
    for (i = 0; i < (N - 1) * (N - 1); ++i){
        rb[i] = i + N * N * 2;
    }
    for (i = 0; i < (N - 1) * N; ++i){
        rc[i] = i + N * N * 3;
    }
    for (i = 0; i < (N - 1) * N; ++i){
        rd[i] = i + N * N * 4;
    }        
    
    wrap_np_matrix(&a, ra, N, N, 'N');
    wrap_np_matrix(&b, rb, N - 1, N - 1, 'N');
    wrap_np_matrix(&c, rc, N, N - 1, 'N');
    wrap_np_matrix(&d, rd, N - 1, N, 'N');
    
    wrap_np_matrix(&f, rf, N * 2 - 1, N * 2 - 1, 'N');
    wrap_np_matrix(&g, rg, N * 2 - 1, N * 2 - 1, 'N');
    
    //test_print_np_matrix(&a);
    //test_print_np_matrix(&b);
    //test_print_np_matrix(&c);
    //test_print_np_matrix(&d);
    double stime = omp_get_wtime();
   // for (i = 0; i < 1; ++i){
   copy_np_matrix(&f, 0, 0, &a);
   copy_np_matrix(&f, a.m, a.n, &b);
   copy_np_matrix(&f, 0, a.n, &c);
   copy_np_matrix(&f, a.m, 0, &d);
   test_print_np_matrix(&f);
   transpose_np_matrix(&f);
   strided_copy_np_matrix(&g, b.n, b.m, &f, a.n, a.m, 0, 0);
   strided_copy_np_matrix(&g, 0, 0, &f, b.n, b.m, a.n, a.m);
   strided_copy_np_matrix(&g, 0, b.m, &f, c.n, c.m, a.n, 0);
   strided_copy_np_matrix(&g, b.n, 0, &f, d.n, d.m, 0, a.m);
   test_print_np_matrix(&g);
   
/*        strided_copy_np_matrix(&f, 0, 0, &a, a.m - 1, a.n - 1, 0, 0);*/
/*        strided_copy_np_matrix(&f, 1, a.n + 1, &b, b.m - 1, b.n - 1, 1, 1);*/
/*        strided_copy_np_matrix(&f, a.m, 0, &c, c.m, c.n, 0, 0);*/
/*        strided_copy_np_matrix(&f, a.m, c.n, &d, d.m, d.n, 0, 0);*/
    //}
    fprintf(stdout, "%lf\n", omp_get_wtime() - stime); 
    
    free(ra);
    free(rb);
    free(rc);
    free(rd);
    free(rf);
    free(rg);
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

        
    delete_np_matrix(a);
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
    axpy_np_matrix(a, a, alpha, a);
    test_print_np_matrix(a);
    
/*    transpose_np_matrix(a);*/
/*    transpose_np_matrix(b);*/
/*    transpose_np_matrix(e);*/
/*    test_print_np_matrix(a);*/
/*    axpy_np_matrix(c, a, alpha, b);*/
/*    test_print_np_matrix(c);*/

/*    test_print_np_matrix(b);*/
/*    test_print_np_matrix(e);*/
/*    axpy_np_matrix(c, b, alpha, e);*/
/*    test_print_np_matrix(c);*/

    
    delete_np_matrix(a);
    delete_np_matrix(b);
    delete_np_matrix(c);    
    delete_np_matrix(d); 
    delete_np_matrix(e); 
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
    
    scal_np_matrix(c, a, alpha);
    test_print_np_matrix(c);
    
    transpose_np_matrix(a);
    test_print_np_matrix(a);
    scal_np_matrix(d, a, alpha);
    test_print_np_matrix(d);
 
    
    delete_np_matrix(a);
    delete_np_matrix(c);    
    delete_np_matrix(d);    
}

void test_ddot_np_matrix(){
    int i;
    NP_MATRIX *a = new_np_matrix(4, 3);
    NP_MATRIX *b = new_np_matrix(4, 2);
    NP_MATRIX *c = new_np_matrix(3, 2);
    for (i = 0; i < 12; ++i){
        a->matrix[i] = i + 1;
    }
    
    for (i = 0; i < 8; ++i){
        b->matrix[i] = i + 3;
    }    
    
    
    //transpose_np_matrix(b);
    transpose_np_matrix(a);
    test_print_np_matrix(a);
    test_print_np_matrix(b);
    
    
    ddot_np_matrix(c, a, b);
    test_print_np_matrix(c);
    
    //transpose_np_matrix(a);
    
    delete_np_matrix(a);
    delete_np_matrix(b);    
    delete_np_matrix(c);  
}

void test_delete_np_matrix(NP_MATRIX **a){
    test_print_np_matrix(*a);
    delete_np_matrix(a);
    test_print_np_matrix(*a);
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

int intbsearch(const int const *arr, const int length, const int x){ 
    int l, m, r;
    l = 0;
    r = length - 1;
    
    if (x > arr[r]){
        return -1;
    }
    
    while (l <= r) { 
        m = l + (r - l) / 2; 
        if (arr[m] == x){
            return m; 
        }
        if (arr[m] < x){
            l = m + 1; 
        }
        else{
            r = m - 1; 
        }
    } 
    return -1; 
} 


int main(){
    logf("Printing\n");
    return 0;
}

