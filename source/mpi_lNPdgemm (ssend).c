#include "mpi_helper.h"   
#include "omp_helper.h" 
static int sg_i = 0;
static int sg_count = 0;
static sem_t i_lock;
static sem_t t_lock;
static sem_t r_lock[WGRAPE_SIZE];
static sem_t p_lock[WGRAPE_SIZE];

static char *sg_tra_a, *sg_tra_b;
static int *sg_m_arr;
static int *sg_n_arr;
static int *sg_k_arr;
static int *sg_lda;
static int *sg_ldb;
static int *sg_ldc;
static int *sg_offseta;
static int *sg_offsetb;
static int *sg_offsetc;
static double *sg_temp_arr;
static double **sg_a_arr;
static double **sg_b_arr;
static double **sg_c_arr;
static double *sg_alpha;
static double *sg_beta;
static int *sg_completed;
static int sg_matrix_num;
extern MPI_Datatype MPI_MATRIX_META;
extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];
extern double *TEMP_ARR;
extern double **A_ARR, **B_ARR, **C_ARR;
static MATRIX_META sg_matrix_meta[WGRAPE_SIZE];

void matrix_set_meta(const char *tra_a, const char *tra_b, 
                     const int *m_arr, const int *n_arr, const int *k_arr,
                     const int *lda,  const int *ldb,  const int *ldc,
                     const int *offseta, const int *offsetb, const int *offsetc,
                     const double *alpha, const double *beta, MATRIX_META *meta){
    meta->tra_a = *tra_a;
    meta->tra_b = *tra_b;  
    meta->m = *m_arr;
    meta->n = *n_arr;
    meta->k = *k_arr;
    meta->lda = *lda;  
    meta->ldb = *ldb;  
    meta->ldc = *ldc;  
    meta->offseta = *offseta;  
    meta->offsetb = *offsetb;  
    meta->offsetc = *offsetc;  
    meta->alpha = *alpha;  
    meta->beta = *beta;                         
}

void matrix_print_meta(MATRIX_META *meta){
    flogf(stderr, "tra_a=%c, tra_b=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, offseta=%d, offsetb=%d, offsetc=%d, alpha=%lf, beta=%lf\n", 
    meta->tra_a,
    meta->tra_b,
    meta->m,
    meta->n,
    meta->k,
    meta->lda,
    meta->ldb,
    meta->ldc,
    meta->offseta,
    meta->offsetb,
    meta->offsetc,
    meta->alpha,
    meta->beta);
}
   
void *wrapper_NPdgemm(void *data){
    int core_id = (long) data;
    //stick_to_core(core_id);
    int i;
    int s_i[CGRAPE_SIZE] = {0};
    int s_count = 0;
    while (TRUE){
        sem_wait(&i_lock);
        while (s_count < CGRAPE_SIZE){
            
            while (sg_i < sg_matrix_num && sg_completed[sg_i] == 1){
                ++sg_i;
            }
                
            if (sg_i >= sg_matrix_num){
                break;
            } else {
                s_i[s_count++] = sg_i++;
            }
        }
        sem_post(&i_lock);
        while (s_count > 0){
            i = s_i[--s_count];
            NPdgemm(sg_tra_a[i], sg_tra_b[i], sg_m_arr[i], sg_n_arr[i], sg_k_arr[i], sg_lda[i], 
                sg_ldb[i], sg_ldc[i], sg_offseta[i], sg_offsetb[i], sg_offsetc[i], sg_a_arr[i],
                sg_b_arr[i], sg_c_arr[i], sg_alpha[i], sg_beta[i]);

        }       
        
        if (sg_i >= sg_matrix_num){
            pthread_exit(NULL); 
        }        
    }
} 

void *wrapper_Driver(void *data){
    int total_count = 0;
    int i, temp_sc, s_count;
    int dest = *(int *)data;
	MATRIX_META matrix_meta[WGRAPE_SIZE];
    int s_i[WGRAPE_SIZE] = {0};
    
    while (TRUE){
        s_count = 0;
        sem_wait(&i_lock);
        
        while (s_count < WGRAPE_SIZE){
            
            while (sg_i < sg_matrix_num && sg_completed[sg_i] == 1){
                ++sg_i;
            }
                
            if (sg_i < sg_matrix_num){
                s_i[s_count] = sg_i;
                ++s_count;
                ++sg_i;
                ++total_count;
            } else{
                break;
            }
        }
        sem_post(&i_lock);
        MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
        if (sg_i >= sg_matrix_num && s_count <= 0){
            flogf(stderr, "wrapper_Driver: dest=%d was distributed %d tasks\n", dest, total_count);
            pthread_exit(NULL); 
        } 
        MPI_Ssend(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
        
        temp_sc = s_count;
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            matrix_set_meta(sg_tra_a + i, sg_tra_b + i, sg_m_arr + i, sg_n_arr + i, 
                            sg_k_arr + i, sg_lda + i, sg_ldb + i, sg_ldc + i, 
                            sg_offseta + i, sg_offsetb + i, sg_offsetc + i, 
                            sg_alpha + i, sg_beta + i, matrix_meta + s_count);
            MPI_Send(matrix_meta + s_count, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * s_count + META_TAG, MPI_COMM_WORLD);
            MPI_Send(sg_a_arr[i], sg_m_arr[i] * sg_k_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_A_TAG, MPI_COMM_WORLD);
            MPI_Send(sg_b_arr[i], sg_k_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_B_TAG, MPI_COMM_WORLD);
        }

        
        s_count = temp_sc;
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            MPI_Recv(sg_c_arr[i], sg_m_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}  

void *wrapper_Receiver(void *data){
    int i, flag, r_count;
    int s_count = 0;
    int dest = 0; // to root
    int notreceived[WGRAPE_SIZE];
	int s_i[WGRAPE_SIZE];
    
    sem_wait(&i_lock);
    MPI_Recv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    s_count = sg_count;
	

	
    sem_post(&i_lock);
    while (s_count > 0){
		MPI_Recv(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = s_count - 1; i >= 0; --i){
            MPI_Recv(sg_matrix_meta + i, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * i + META_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(sg_a_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].k, MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_A_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);         
            MPI_Recv(sg_b_arr[i], sg_matrix_meta[i].k * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_B_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sem_post(&t_lock);   
            sem_post(r_lock + i); 
        }

        for (i = s_count - 1; i >= 0; --i){
            sem_wait(p_lock + i);
            MPI_Ssend(sg_c_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD);
        }
        sem_wait(&i_lock);
        MPI_Recv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        s_count = sg_count;     
        sem_post(&i_lock);
    }       
    sem_wait(&i_lock);
    sg_count = -1;
    sem_post(&i_lock);
    for (i = 0; i < NUM_OF_WTHREADS; i++){
        sem_post(&t_lock);
    }
    pthread_exit(NULL); 
}

void *wrapper_Worker(void *data){
    int core_id = (long) data;
    int i;
    while (TRUE){
        sem_wait(&t_lock);
        sem_wait(&i_lock);
        if (sg_count > 0){
            --sg_count;
            i = sg_count;
            sem_post(&i_lock);
            sem_wait(r_lock + i);
            NPdgemm(sg_matrix_meta[i].tra_a, sg_matrix_meta[i].tra_b,
                sg_matrix_meta[i].m, sg_matrix_meta[i].n, sg_matrix_meta[i].k,
                sg_matrix_meta[i].lda, sg_matrix_meta[i].ldb, sg_matrix_meta[i].ldc,
                sg_matrix_meta[i].offseta, sg_matrix_meta[i].offsetb,                      
                sg_matrix_meta[i].offsetc, sg_a_arr[i], sg_b_arr[i], sg_c_arr[i],
                sg_matrix_meta[i].alpha, sg_matrix_meta[i].beta); 
            sem_post(p_lock + i); 
        } else if (sg_count < 0){
            sem_post(&i_lock);
            break;
        } else {
            sem_post(&i_lock);
        }
         
        
    }
    pthread_exit(NULL); 
}

void mpi_lNPdgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num, int * completed){    
    double stime = omp_get_wtime();                 
    int signal = SIGNAL_LNPDGEMM;     
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }    
    int i, *dest;
    sem_init(&i_lock, 0, 1);
    //Root:
    if (WORLD_RANK == 0){ 
        
        sg_tra_a = tra_a;
        sg_tra_b = tra_b;
        sg_m_arr = m_arr;
        sg_n_arr = n_arr;
        sg_k_arr = k_arr;
        sg_lda = lda;
        sg_ldb = ldb;
        sg_ldc = ldc;
        sg_offseta = offseta;
        sg_offsetb = offsetb;
        sg_offsetc = offsetc;
        sg_a_arr = a_arr;
        sg_b_arr = b_arr;
        sg_c_arr = c_arr;
        sg_alpha = alpha;
        sg_beta = beta;
        sg_completed = completed;
        sg_matrix_num = matrix_num; 
        sg_i = 0;   

        dest = malloc(sizeof(int) * (WORLD_SIZE - 1));
        pthread_t *driving_threads = malloc(sizeof(pthread_t) * NUM_OF_WTHREADS * (WORLD_SIZE - 1));
        pthread_t computing_threads[NUM_OF_CTHREADS];            
 
        fprintf(stderr, "mpi_lNPdgemm: total %d tasks\n", sg_matrix_num);
        for (i = 0; i < NUM_OF_CTHREADS; ++i){
            pthread_create(computing_threads + i, NULL, wrapper_NPdgemm, (void*) i);
        }
        
        if (NUM_OF_WTHREADS > 0){
            for (i = 0; i < WORLD_SIZE - 1; ++i){
                dest[i] = i + 1;
                pthread_create(driving_threads + i, NULL, wrapper_Driver, dest + i);
            }
        }
        
        for (i = 0; i < NUM_OF_CTHREADS; ++i){
            pthread_join(computing_threads[i], NULL);
        }
	    flogf(stderr, "mpi_lNPdgemm: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
	    if (NUM_OF_WTHREADS > 0){
            for (i = 0; i < WORLD_SIZE - 1; ++i){
                pthread_join(driving_threads[i], NULL);
            }
        }        
 	    
        free(driving_threads);
        free(dest);
        
    }else{
        //Worker:  
        //malloc_stats();
        sem_init(&t_lock, 0, 0); //task lock : how many unfinished task in current comm
        sg_i = 0;
        sg_temp_arr = TEMP_ARR;
        sg_a_arr = A_ARR;
        sg_b_arr = B_ARR;
        sg_c_arr = C_ARR;
        for (i = 0; i < WGRAPE_SIZE; ++i){
            sem_init(r_lock + i, 0, 0); //resource lock : if current task resource is ready
            sem_init(p_lock + i, 0, 0); //product lock : if current product is ready
        }
        

        
        pthread_t working_threads[NUM_OF_WTHREADS];
        pthread_t receiving_thread;
        pthread_create(&receiving_thread, NULL, wrapper_Receiver, NULL);
        for (i = 0; i < NUM_OF_WTHREADS; ++i){
            pthread_create(working_threads + i, NULL, wrapper_Worker, (void*) i);
        }
        
        pthread_join(receiving_thread, NULL);

        for (i = 0; i < NUM_OF_WTHREADS; ++i){
            pthread_join(working_threads[i], NULL);
        }
        
        for (i = 0; i < WGRAPE_SIZE; ++i){
            sem_destroy(r_lock + i);
            sem_destroy(p_lock + i);
        }
        sem_destroy(&t_lock);
    } 
    sem_destroy(&i_lock);
    flogf(stderr, "mpi_lNPdgemm: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
    
}

void pth_lNPdgemm(char * tra_a, char * tra_b,
                  int * m_arr,  int * n_arr, int * k_arr,
                  int * lda,  int * ldb,  int * ldc,
                  int *  offseta,  int *  offsetb, int *  offsetc,
                  double * * a_arr, double * * b_arr, double * * c_arr,
                  double *  alpha,  double *  beta,
                  int matrix_num, int * completed){    
    double stime = omp_get_wtime();                 
    int i;
    sem_init(&i_lock, 0, 1);
    sg_tra_a = tra_a;
    sg_tra_b = tra_b;
    sg_m_arr = m_arr;
    sg_n_arr = n_arr;
    sg_k_arr = k_arr;
    sg_lda = lda;
    sg_ldb = ldb;
    sg_ldc = ldc;
    sg_offseta = offseta;
    sg_offsetb = offsetb;
    sg_offsetc = offsetc;
    sg_a_arr = a_arr;
    sg_b_arr = b_arr;
    sg_c_arr = c_arr;
    sg_alpha = alpha;
    sg_beta = beta;
    sg_completed = completed;
    sg_matrix_num = matrix_num; 
    sg_i = 0;   
    pthread_t computing_threads[NUM_OF_CTHREADS];            
        
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_create(computing_threads + i, NULL, wrapper_NPdgemm, NULL);
    }
    
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_join(computing_threads[i], NULL);
    }
 	    
    sem_destroy(&i_lock);
    flogf(stderr, "pth_lNPdgemm: %f seconds taken.\n", omp_get_wtime() - stime);
}
