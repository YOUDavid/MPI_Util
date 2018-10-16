#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <complex.h>
#include <omp.h>
#include <mpi.h>
//#include <cblas.h>
#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))



int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getMem(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}





void dgemm_(const char *, const char *,
            const int *, const int *, const int *,
            const double *, const double *, const int *,
            const double *, const int *,
            const double *, double *, const int *);

void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta);
   
void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta){
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

#pragma omp parallel default(none) shared(a, b, c) \
        private(i, j)
      {
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
  else
    {                // parallelize n

#pragma omp parallel default(none) shared(a, b, c)
      {
    int nthread = omp_get_num_threads ();
    int nblk = MAX ((n + nthread - 1) / nthread, 1);
    nthread = (n + nblk - 1) / nblk;
    int di;
    size_t bstride = nblk;
    size_t cstride = dimc * nblk;
    if (trans_b == 'N')
      {
        bstride *= ldb;
      }
#pragma omp for
    for (i = 0; i < nthread; i++)
      {
        di = MIN (nblk, n - i * nblk);
        if (di > 0)
          {
        dgemm_ (&trans_a, &trans_b, &m, &di, &k,
            &alpha, a, &lda, b + bstride * i, &ldb,
            &beta, c + cstride * i, &ldc);
          }
      }
      }
    }
}
   
             
void mpi_lNPdgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num){  
	struct timeval start, stop;
	gettimeofday(&start, NULL);
	int mem = getMem();
	
	
    int signal = 0; //0 for mpi_lNPdgemm            
    int world_size;
    int world_rank;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
	
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //fprintf(stderr, "P%d: Starting, currently using %d kB\n", world_rank, mem);
    MPI_Get_processor_name(processor_name, &name_len);
    
    if (world_rank == 0){
		fprintf(stderr, "P%d: Starting, signal other nodes\n", world_rank);
		fprintf(stderr, "P%d: &tra_a: %ld, &tra_b:%ld\n", world_rank, tra_a, tra_b);
		fprintf(stderr, "P%d: &m_arr: %ld, &n_arr:%ld, &n_arr:%ld\n", world_rank, m_arr, n_arr, k_arr);
		fprintf(stderr, "P%d: &lda: %ld, &ldb:%ld, &ldc:%ld\n", world_rank, lda, ldb, ldc);
		fprintf(stderr, "P%d: &offseta: %ld, &offsetb:%ld, &offsetc:%ld\n", world_rank, offseta, offsetb, offsetc);
		fprintf(stderr, "P%d: &a_arr: %ld, &b_arr:%ld, &c_arr:%ld\n", world_rank, a_arr, b_arr, c_arr);
		fprintf(stderr, "P%d: &alpha: %ld, &beta:%ld\n", world_rank, alpha, beta);
		fprintf(stderr, "P%d: matrix_num: %d\n", world_rank, matrix_num);
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }                   
    
    MPI_Bcast(&matrix_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int i, rootcount, dest, ind;
    int recvcount = matrix_num / world_size;
    int extracount = matrix_num % world_size;
	int summation = 0;
	
    int *counts = malloc(sizeof(int) * world_size);  
    int *displs = malloc(sizeof(int) * world_size);
	int *a_sizes, *b_sizes, *c_sizes;
	double *temp_a, *temp_b, *temp_c;
	long a_totsiz = 0;
	long b_totsiz = 0;
	long c_totsiz = 0;
	MPI_Request *tot_status;
	
    for (i = 0; i < world_size; i++){
        counts[i] = recvcount;
        if (extracount > 0){
            counts[i]++;
            extracount--;
        }
        displs[i] = summation;
        summation += counts[i];
    }
	

    rootcount = counts[0];
    counts[0] = 0;    
    
    if (world_rank != 0){
        tra_a = malloc(sizeof(char) * counts[world_rank]);
        tra_b = malloc(sizeof(char) * counts[world_rank]);
        m_arr = malloc(sizeof(int) * counts[world_rank]);
        n_arr = malloc(sizeof(int) * counts[world_rank]);
        k_arr = malloc(sizeof(int) * counts[world_rank]);    
        lda = malloc(sizeof(int) * counts[world_rank]); 
        ldb = malloc(sizeof(int) * counts[world_rank]); 
        ldc = malloc(sizeof(int) * counts[world_rank]); 
        offseta = malloc(sizeof(int) * counts[world_rank]); 
        offsetb = malloc(sizeof(int) * counts[world_rank]); 
        offsetc = malloc(sizeof(int) * counts[world_rank]); 
        a_arr = malloc(sizeof(double*) * counts[world_rank]);  
        b_arr = malloc(sizeof(double*) * counts[world_rank]);  
        c_arr = malloc(sizeof(double*) * counts[world_rank]);
        alpha = malloc(sizeof(double*) * counts[world_rank]);  
        beta = malloc(sizeof(double*) * counts[world_rank]);          
		a_sizes = malloc(sizeof(int) * counts[world_rank]);
		b_sizes = malloc(sizeof(int) * counts[world_rank]);
		c_sizes = malloc(sizeof(int) * counts[world_rank]);     
    } else {
		a_sizes = malloc(sizeof(int) * (matrix_num - rootcount));
		b_sizes = malloc(sizeof(int) * (matrix_num - rootcount));
		c_sizes = malloc(sizeof(int) * (matrix_num - rootcount));
	}
	
    MPI_Scatterv(tra_a, counts, displs, MPI_CHAR, tra_a, counts[world_rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(tra_b, counts, displs, MPI_CHAR, tra_b, counts[world_rank], MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(m_arr, counts, displs, MPI_INT, m_arr, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);    
    MPI_Scatterv(n_arr, counts, displs, MPI_INT, n_arr, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(k_arr, counts, displs, MPI_INT, k_arr, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(lda, counts, displs, MPI_INT, lda, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ldb, counts, displs, MPI_INT, ldb, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ldc, counts, displs, MPI_INT, ldc, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(offseta, counts, displs, MPI_INT, offseta, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(offsetb, counts, displs, MPI_INT, offsetb, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(offsetc, counts, displs, MPI_INT, offsetc, counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(alpha, counts, displs, MPI_DOUBLE, alpha, counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(beta, counts, displs, MPI_DOUBLE, beta, counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	
	if (world_rank != 0){
		//#pragma omp parallel for
		for (i = 0; i < counts[world_rank]; i++){
		    //fprintf(stderr, "P%d: Adding a_sizes[%d]:%d to a_totsiz:%ld\n", world_rank, i, a_sizes[i], a_totsiz);
			a_sizes[i] = m_arr[i] * k_arr[i];
			a_totsiz += a_sizes[i];
			b_sizes[i] = k_arr[i] * n_arr[i];
			b_totsiz += b_sizes[i];
			c_sizes[i] = n_arr[i] * m_arr[i];
			c_totsiz += c_sizes[i];
		}	
		
		temp_a = malloc(sizeof(double) * a_totsiz);    
		temp_b = malloc(sizeof(double) * b_totsiz); 
		temp_c = malloc(sizeof(double) * c_totsiz); 
	    if (temp_a == NULL){
	        fprintf(stderr, "P%d: temp_a is NULL receiving buffer! a_totsiz:%ld\n", world_rank, a_totsiz);
			//signal root
			signal = -1;
	    } 
	    if (temp_b == NULL){
	        fprintf(stderr, "P%d: temp_b is NULL receiving buffer! &b_totsiz:%ld\n", world_rank, b_totsiz);
			signal = -1;
	    } 
	    if (temp_c == NULL){
	        fprintf(stderr, "P%d: temp_c is NULL receiving buffer! &c_totsiz:%ld\n", world_rank, c_totsiz);
	        signal = -1;
	    } else{
	        signal = 0;
	    }
		
		/*MPI_Isend(&signal, 1, MPI_INT, 0, world_rank, MPI_COMM_WORLD, NULL);
		MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (signal < 0){
			free(tra_a);
            free(tra_b);
            free(m_arr);
            free(n_arr);
            free(k_arr);
            free(lda);
            free(ldb);
            free(ldc);
            free(offseta);
            free(offsetb);
            free(offsetc);
            free(a_arr);
            free(b_arr);
            free(c_arr);
            free(a_sizes);
            free(b_sizes);
            free(c_sizes);
            free(alpha);
            free(beta); 
			free(counts);
			free(displs); 
			return;
		}*/
		
		a_arr[0] = temp_a;
		b_arr[0] = temp_b;
		c_arr[0] = temp_c;
		//#pragma omp parallel for
		for (i = 1; i < counts[world_rank]; i++){
			a_arr[i] = a_arr[i - 1] + a_sizes[i - 1];
			b_arr[i] = b_arr[i - 1] + b_sizes[i - 1];
			c_arr[i] = c_arr[i - 1] + c_sizes[i - 1];
		}	
		
		
		tot_status = malloc(sizeof(MPI_Request) * 2 * counts[world_rank]);
		//init temp a, b
		for (i = 0; i < counts[world_rank]; i++){
		    /*if (a_arr[i] == NULL || b_arr[i] == NULL){
		        fprintf(stderr, "P%d: NULL receiving buffer!\n", world_rank);
		    }*/
			MPI_Irecv(a_arr[i], a_sizes[i], MPI_DOUBLE, 0, i * 2, MPI_COMM_WORLD, tot_status + i * 2);
			MPI_Irecv(b_arr[i], b_sizes[i], MPI_DOUBLE, 0, i * 2 + 1, MPI_COMM_WORLD, tot_status + i * 2 + 1);
		}	
		
		MPI_Waitall(counts[world_rank] * 2, tot_status, MPI_STATUSES_IGNORE);
		
	} else{
		//#pragma omp parallel for
		for (i = rootcount; i < matrix_num; i++){
			a_sizes[i - rootcount] = m_arr[i] * k_arr[i];
			b_sizes[i - rootcount] = k_arr[i] * n_arr[i];
			c_sizes[i - rootcount] = n_arr[i] * m_arr[i];
		}
		
		//Check malloc on workers
		/*signal = 0;
		for (i = 1; i < world_size; i++){
			MPI_Recv(&signal, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (signal < 0){
				break;
			}
		}*/
		
		/*MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (signal < 0){
			free(counts);
			free(displs);  
			free(a_sizes);
			free(b_sizes);
			free(c_sizes);
			fprintf(stderr, "Lddot failed because of insufficient memory!\n");
			return;
		}*/
		
		//Send out arr a, b to temp a, b
		tot_status = malloc(sizeof(MPI_Request) * 2 * (matrix_num - rootcount));	
		for (dest = 1; dest < world_size; dest++){
			for (ind = 0; ind < counts[dest]; ind++){
				i = ind + displs[dest];
				MPI_Isend(a_arr[i], a_sizes[i - rootcount], MPI_DOUBLE, dest, ind * 2, MPI_COMM_WORLD, tot_status + (i - rootcount) * 2);
				MPI_Isend(b_arr[i], b_sizes[i - rootcount], MPI_DOUBLE, dest, ind * 2 + 1, MPI_COMM_WORLD, tot_status + (i - rootcount) * 2 + 1);
				//fprintf(stderr, "Sending out a[%d][0]=%f, b[%d][0]=%f to %d\n", i, a_arr[i][0], i, b_arr[i][0], dest);
			}
		}
	}    

    counts[0] = rootcount;    
    for (i = 0; i < counts[world_rank]; i++){
        //fprintf(stderr, "P%d: a[%d][0]=%f, b[%d][0]=%f\n",world_rank, i, a_arr[i][0], i, b_arr[i][0]);
        NPdgemm(tra_a[i], tra_b[i], m_arr[i], n_arr[i], k_arr[i], lda[i], 
                ldb[i], ldc[i], offseta[i], offsetb[i], offsetc[i], a_arr[i],
                b_arr[i], c_arr[i], alpha[i], beta[i]);
    }
	counts[0] = 0;
	
	if (world_rank != 0){
		for (i = 0; i < counts[world_rank]; i++){
			MPI_Isend(c_arr[i], c_sizes[i], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, tot_status + i);
		}	
		
		/*for (i = 0; i < counts[world_rank]; i++){
			MPI_Wait(tot_status + i, MPI_STATUS_IGNORE);
		}*/
		MPI_Waitall(counts[world_rank], tot_status, MPI_STATUSES_IGNORE);
		
	} else{
		/*for (i = 0; i < matrix_num - rootcount; i++){
			MPI_Wait(tot_status + i * 2, MPI_STATUS_IGNORE);
			MPI_Wait(tot_status + i * 2 + 1, MPI_STATUS_IGNORE);
		}*/
		MPI_Waitall((matrix_num - rootcount) * 2, tot_status, MPI_STATUSES_IGNORE);
		
		for (dest = 1; dest < world_size; dest++){
			for (ind = 0; ind < counts[dest]; ind++){
				i = ind + displs[dest];
				MPI_Irecv(c_arr[i], c_sizes[i - rootcount], MPI_DOUBLE, dest, ind, MPI_COMM_WORLD, tot_status + i - rootcount);
			}
		}
		
		/*for (i = 0; i < matrix_num - rootcount; i++){
			MPI_Wait(tot_status + i, MPI_STATUS_IGNORE);
		}*/
		MPI_Waitall(matrix_num - rootcount, tot_status, MPI_STATUSES_IGNORE);
	}
	

    
	gettimeofday(&stop, NULL); 
	fprintf(stderr, "%f seconds taken from processor %s, rank %d out of %d worlds.\n", 
	       (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec),
		   processor_name, world_rank, world_size);	
	MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0){
        free(tra_a);
        free(tra_b);
        free(m_arr);
        free(n_arr);
        free(k_arr);
        free(lda);
        free(ldb);
        free(ldc);
        free(offseta);
        free(offsetb);
        free(offsetc);
        free(a_arr);
        free(b_arr);
        free(c_arr);
        free(temp_a);
        free(temp_b);
        free(temp_c);
        free(alpha);
        free(beta);
    }
    free(counts);
    free(displs);  
	free(a_sizes);
	free(b_sizes);
	free(c_sizes);
	free(tot_status);
	mem = getMem();
	//fprintf(stderr, "P%d: Finishing, currently using %d kB\n", world_rank, mem);
} 

void mpi_print(int len, double* arr) {
    int signal = 1; //1 for mpi_print
    int world_size;
    int world_rank;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Get_processor_name(processor_name, &name_len);
    if (world_rank == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    int i;
    int mpi_status;
    int recvcount;
    int extracount;
    int *counts;
    int rootcount;
    int *displs;
    int summation = 0;
    
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    recvcount = len / world_size;
    extracount = len % world_size;  
    counts = malloc(sizeof(int) * world_size);

    displs = malloc(sizeof(int) * world_size);
    for (i = 0; i < world_size; i++){
        counts[i] = recvcount;
        if (extracount > 0){
            counts[i]++;
            extracount--;
        }
        displs[i] = summation;
        summation += counts[i];
    }
    rootcount = counts[0];
    counts[0] = 0;
    
    if (world_rank != 0){
        arr = malloc(sizeof(double) * counts[world_rank]);
    }
    mpi_status = MPI_Scatterv(arr, counts, displs, MPI_DOUBLE, arr, 
                                  counts[world_rank], MPI_DOUBLE, 0, 
                                  MPI_COMM_WORLD);
    counts[0] = rootcount;                             
    #pragma omp parallel for
    for (i = 0; i < counts[world_rank]; i++){
      printf("Number %f from cpu %3d in processor %s, rank %d out of %d processors/worlds \n", arr[i], sched_getcpu(), processor_name, world_rank, world_size);
      arr[i]++;
    }
    counts[0] = 0;

 
    mpi_status = MPI_Gatherv(arr, counts[world_rank], MPI_DOUBLE, arr,
                                 counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                                 
    if (world_rank != 0){
        free(arr);
    }                                 
    free(counts);
    free(displs);
}

void mpi_init(){
    int signal; //chose which function to execute
    int world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    fprintf(stderr, "P%d: Initialized\n", world_rank);
    if (world_rank != 0){
        //In worker processes, block until root signals workers to continue
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        while (signal >= 0){
            switch (signal){
                case 1:
                    mpi_print(-1, NULL);     
                break;
                case 0:
                    mpi_lNPdgemm(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 0);
            }
            //After finishing a task, block again
            MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        fprintf(stderr, "P%d: Finalized\n", world_rank);
        MPI_Finalize();
    } else{
        //In root process, do nothing, just return
    }
}


void mpi_final(){
    int signal = -1;
    MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    fprintf(stderr, "P%d: Finalized\n", 0);
    MPI_Finalize();
}








