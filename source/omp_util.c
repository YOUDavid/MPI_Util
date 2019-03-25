#include "omp_helper.h"

int stick_to_core(int core_id) {
/*   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);*/
/*   if (core_id < 0 || core_id >= num_cores)*/
/*      return EINVAL;*/

/*   cpu_set_t cpuset;*/
/*   CPU_ZERO(&cpuset);*/
/*   CPU_SET(core_id, &cpuset);*/

/*   pthread_t current_thread = pthread_self();    */
/*   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);*/
    return 0;
}

void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta){
/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */   

  omp_set_num_threads(atoi(getenv("OMP_NUM_THREADS")));
  const size_t dimc = ldc;
  int i, j;
  if (m == 0 || n == 0){
    return;
  }
  else if (k == 0){
    for (i = 0; i < n; i++){
      for (j = 0; j < m; j++){
          c[i * dimc + j] = 0;
      }
    }
    return;
  }
  a += offseta;
  b += offsetb;
  c += offsetc;
  
  if ((k / m) > 3 && (k / n) > 3){ 
  // parallelize k
    if (beta == 0){
      for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
          c[i * dimc + j] = 0;
        }
      }
    }else{
      for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
          c[i * dimc + j] *= beta;
        }
      }
    }
  
#pragma omp parallel default(none) shared(a, b, c) private(i, j)
      {
    //printf("OMP NPdgemm using %d threads.\n", omp_get_num_threads());
    int nthread = omp_get_num_threads ();
    int nblk = MAX ((k + nthread - 1) / nthread, 1);
    double D0 = 0;
    double *cpriv = malloc (sizeof (double) * (m * n + 2));
    int di;
    size_t ij;
    size_t astride = nblk;
    size_t bstride = nblk;
    if (trans_a == 'N')
      {
        astride *= lda;
      }
    if (trans_b != 'N')
      {
        bstride *= ldb;
      }
#pragma omp for
    for (i = 0; i < nthread; i++)
      {
        di = MIN (nblk, k - i * nblk);
        if (di > 0)
          {
        dgemm_ (&trans_a, &trans_b, &m, &n, &di,
            &alpha, a + astride * i, &lda,
            b + bstride * i, &ldb, &D0, cpriv, &m);
          }
      }
#pragma omp critical
    if (di > 0)
      {
        for (ij = 0, i = 0; i < n; i++)
          {
          
        for (j = 0; j < m; j++, ij++)
          {
            c[i * dimc + j] += cpriv[ij];
          }
          }
      }
    free (cpriv);
      }

    }
  else if (m > n * 2)
    {                // parallelize m

#pragma omp parallel default(none) shared(a, b, c)
      {
    int nthread = omp_get_num_threads ();
    int nblk = MAX ((m + nthread - 1) / nthread, 1);
    nthread = (m + nblk - 1) / nblk;
    int di;
    size_t bstride = nblk;
    if (trans_a != 'N')
      {
        bstride *= lda;
      }
#pragma omp for
    for (i = 0; i < nthread; i++)
      {
        di = MIN (nblk, m - i * nblk);
        if (di > 0)
          {
        dgemm_ (&trans_a, &trans_b, &di, &n, &k,
            &alpha, a + bstride * i, &lda, b, &ldb,
            &beta, c + i * nblk, &ldc);
          }
      }
      }

    }
  else{                // parallelize n
    #pragma omp parallel default(none) shared(a, b, c)
    {
      int nthread = omp_get_num_threads();
      int nblk = MAX ((n + nthread - 1) / nthread, 1);
      nthread = (n + nblk - 1) / nblk;
      int di;
      size_t bstride = nblk;
      size_t cstride = dimc * nblk;
      if (trans_b == 'N'){
        bstride *= ldb;
      }
      #pragma omp for
      for (i = 0; i < nthread; i++){
        di = MIN (nblk, n - i * nblk);
        if (di > 0){
          dgemm_ (&trans_a, &trans_b, &m, &di, &k,
                  &alpha, a, &lda, b + bstride * i, &ldb,
                  &beta, c + cstride * i, &ldc);
        }
      }
    }
  }
}

void omp_dgemm(const charptr RESTRICT tra_a, const charptr RESTRICT tra_b,
               const intptr RESTRICT m_arr, const intptr RESTRICT n_arr, const intptr RESTRICT k_arr,
               const doubleptr RESTRICT alpha, const doubleptrptr RESTRICT a_arr, const intptr RESTRICT lda,
               const doubleptrptr RESTRICT b_arr, const intptr RESTRICT ldb, const doubleptr RESTRICT beta,
               doubleptrptr RESTRICT c_arr, const intptr  RESTRICT ldc,
               const RESTRICT intptr completed, const int num){
    int i;
    #pragma omp parallel for
    for (i = 0; i < num; ++i){
        if (completed[i] == 0){
            dgemm_(tra_a + i, tra_b + i, m_arr + i, n_arr + i, k_arr + i,
                   alpha + i, a_arr[i], lda + i, b_arr[i], ldb + i, beta + i,
                   c_arr[i], ldc + i);
        }
    }
}


int omp_sum_int(const intptr RESTRICT arr, const int num){
    int i;
    int sum = 0;
    #pragma omp parallel for reduction (+:sum)
    for(i = 0; i < num; ++i) {
        sum = sum + arr[i];
    }
    return sum;
}    

void omp_arr_mul_int(intptr RESTRICT c, const intptr RESTRICT a, const intptr RESTRICT b, const int num){
    int i;
    #pragma omp parallel for //reduction (+:sum)
    for(i = 0; i < num; ++i) {
        c[i] = a[i] * b[i];
    }
}        

void omp_arr_ofs_dou(doubleptrptr RESTRICT arr, const intptr RESTRICT ofs, const int num){
    //#pragma omp parallel for
    //for (i = 1; i < counts[world_rank]; i++){
    //	a_arr[i] = a_arr[i - 1] + a_sizes[i - 1];
    //}	
    int i;
    #pragma omp parallel for
    for (i = 1; i < num; ++i){
        arr[i] = arr[i - 1] + ofs[i - 1];       
    }
}


void omp_print(int len, double* arr) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < len; i++){
      printf("Number %f from cpu %3d \n", arr[i], sched_getcpu());
      arr[i]++;
    }
}





