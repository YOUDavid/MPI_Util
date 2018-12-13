#include "mpi_helper.h"   
#include "omp_helper.h" 
#define DEBUG 0
static int sg_i = 0;
static int sg_count = 0;
//static int sg_notreceived[WGRAPE_SIZE];
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
//extern MPI_Comm *WTHREAD_COMM;
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
    fprintf(stderr, "tra_a=%c, tra_b=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, offseta=%d, offsetb=%d, offsetc=%d, alpha=%lf, beta=%lf\n", 
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
                sem_post(&i_lock);
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
    MPI_Request meta_status[WGRAPE_SIZE], data_a_status[WGRAPE_SIZE], data_b_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE], signal_status;
    MATRIX_META matrix_meta[WGRAPE_SIZE];
    int s_i[WGRAPE_SIZE] = {0};
    //fprintf(stderr, "wrapper_Driver: total %d tasks\n", sg_matrix_num);
    while (TRUE){
        s_count = 0;
        //fprintf(stderr, "wrapper_Driver: want ilock\n");
        sem_wait(&i_lock);
        //fprintf(stderr, "wrapper_Driver: got ilock\n");
        
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
        if (sg_i >= sg_matrix_num && s_count <= 0){
            sem_post(&i_lock);
            //fprintf(stderr, "wrapper_Driver: release ilock\n");
            fprintf(stderr, "wrapper_Driver: dest=%d was distributed %d tasks\n", dest, total_count);
            //MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
            MPI_Isend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
            MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
            pthread_exit(NULL); 
        } 
        sem_post(&i_lock);
        //fprintf(stderr, "wrapper_Driver: release ilock\n");
        //fprintf(stderr, "wrapper_Driver: s_count=%d\n", s_count);
        //MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
        MPI_Isend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
        
        MPI_Isend(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
        
        temp_sc = s_count;
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            matrix_set_meta(sg_tra_a + i, sg_tra_b + i, sg_m_arr + i, sg_n_arr + i, 
                            sg_k_arr + i, sg_lda + i, sg_ldb + i, sg_ldc + i, 
                            sg_offseta + i, sg_offsetb + i, sg_offsetc + i, 
                            sg_alpha + i, sg_beta + i, matrix_meta + s_count);
			//fprintf(stderr, "wrapper_Driver: sending meta[%d] of global ID = %d of %d X %d to dest=%d with tag=%d\n", s_count, i, sg_m_arr[i], sg_n_arr[i], dest, NUM_OF_TAGS * i + META_TAG);
            MPI_Isend(matrix_meta + s_count, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * i + META_TAG, MPI_COMM_WORLD, meta_status + s_count);
			//fprintf(stderr, "wrapper_Driver: sent meta[%d] of global ID = %d of %d X %d to dest=%d with tag=%d\n", s_count, i, sg_m_arr[i], sg_n_arr[i], dest, NUM_OF_TAGS * i + META_TAG);
            //MPI_Ssend(matrix_meta + s_count, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * s_count + META_TAG, MPI_COMM_WORLD);
            
/*            MPI_Isend(sg_a_arr[i], sg_m_arr[i] * sg_k_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_A_TAG, MPI_COMM_WORLD, data_a_status + s_count);*/
/*            MPI_Isend(sg_b_arr[i], sg_k_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_B_TAG, MPI_COMM_WORLD, data_b_status + s_count);*/
/*            MPI_Irecv(sg_c_arr[i], sg_m_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + RESULT_TAG, MPI_COMM_WORLD, result_status + s_count);*/
        }
        s_count = temp_sc;
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            MPI_Isend(sg_a_arr[i], sg_m_arr[i] * sg_k_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_A_TAG, MPI_COMM_WORLD, data_a_status + s_count);
            MPI_Isend(sg_b_arr[i], sg_k_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_B_TAG, MPI_COMM_WORLD, data_b_status + s_count);
            //MPI_Ssend(sg_a_arr[i], sg_m_arr[i] * sg_k_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_A_TAG, MPI_COMM_WORLD);
            
            /*if (i == 0){
                fprintf(stderr, "wrapper_Driver: a arr error number:%d\n", err);
                fprintf(stderr, "wrapper_Driver: a arr address:%lu\n", sg_a_arr[i]);
                fprintf(stderr, "wrapper_Driver: a arr with size:%d\n", sg_k_arr[i] * sg_n_arr[i]);
                test_print_arr(sg_a_arr[i],  sg_m_arr[i], sg_k_arr[i]);
            }*/
            //MPI_Ssend(sg_b_arr[i], sg_k_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + DATA_B_TAG, MPI_COMM_WORLD);
        }
        s_count = temp_sc;
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
			//fprintf(stderr, "wrapper_Driver: expecting result[%d] of global ID = %d of %d X %d from src=%d and tag=%d\n", s_count, i, sg_m_arr[i], sg_n_arr[i], dest, NUM_OF_TAGS * i + RESULT_TAG);
            MPI_Irecv(sg_c_arr[i], sg_m_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD, result_status + s_count);
            //MPI_Recv(sg_c_arr[i], sg_m_arr[i] * sg_n_arr[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //fprintf(stderr, "wrapper_Driver: waiting for meta\n");
        //MPI_Waitall(temp_sc, meta_status, MPI_STATUSES_IGNORE);
        //fprintf(stderr, "wrapper_Driver: waiting for a arrs\n");
        //MPI_Waitall(temp_sc, data_a_status, MPI_STATUSES_IGNORE);
        //fprintf(stderr, "wrapper_Driver: waiting for b arrs\n");
        //MPI_Waitall(temp_sc, data_b_status, MPI_STATUSES_IGNORE);
        //fprintf(stderr, "wrapper_Driver: waiting for result\n");
        MPI_Waitall(temp_sc, result_status, MPI_STATUSES_IGNORE);
        //fprintf(stderr, "wrapper_Driver: all received\n");
    }
}  

void *wrapper_Receiver(void *data){
    int i, flag, r_count;
    int s_count = 0;
    int dest = 0; // to root
    int notreceived[WGRAPE_SIZE];
    int s_i[WGRAPE_SIZE];
    MPI_Request meta_status[WGRAPE_SIZE], data_a_status[WGRAPE_SIZE], data_b_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE], signal_status;
    
    
    //fprintf(stderr, "wrapper_Receiver: want ilock\n");
    sem_wait(&i_lock);
    //fprintf(stderr, "wrapper_Receiver: got ilock\n");
    MPI_Irecv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
    MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
    s_count = sg_count;

    sem_post(&i_lock);
    //fprintf(stderr, "wrapper_Receiver: release ilock\n");
/*    for (i = s_count - 1; i >= 0; --i){*/
/*        fprintf(stderr, "wrapper_Receiver: want wlock:%d\n", i);*/
/*        sem_wait(w_lock + i);*/
/*        fprintf(stderr, "wrapper_Receiver: got wlock:%d\n", i);*/
/*    }*/
    
    //int printed = 0;
    while (s_count > 0){
		
		MPI_Irecv(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
		MPI_Wait(&signal_status, MPI_STATUS_IGNORE);	
		
        for (i = s_count - 1; i >= 0; --i){
	    //fprintf(stderr, "wrapper_Receiver: receiving meta[%d] of global ID = %d from root with tag=%d\n", i, s_i[i], NUM_OF_TAGS * s_i[i] + META_TAG);
            MPI_Irecv(sg_matrix_meta + i, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * s_i[i] + META_TAG, MPI_COMM_WORLD, meta_status + i);
			
            //MPI_Recv(sg_matrix_meta + i, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * i + META_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sem_post(&t_lock);    
            //fprintf(stderr, "wrapper_Receiver: release tlock\n");  
        }
        //fprintf(stderr, "wrapper_Receiver: Received all meta, s_count=%d\n", s_count);
        for (i = s_count - 1; i >= 0; --i){
            MPI_Wait(meta_status + i, MPI_STATUS_IGNORE);
	    //fprintf(stderr, "wrapper_Receiver: received meta[%d] of global ID = %d of %d X %d from root with tag=%d\n", i, s_i[i], sg_matrix_meta[i].m, sg_matrix_meta[i].n, NUM_OF_TAGS * s_i[i] + META_TAG);
            notreceived[i] = 2;
            MPI_Irecv(sg_a_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].k, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + DATA_A_TAG, MPI_COMM_WORLD, data_a_status + i);              
            MPI_Irecv(sg_b_arr[i], sg_matrix_meta[i].k * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + DATA_B_TAG, MPI_COMM_WORLD, data_b_status + i);
            //int err = MPI_Recv(sg_a_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].k, MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_A_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   
            
            /*if (i == 0 && printed == 0){
                //fprintf(stderr, "wrapper_Receiver: a arr error number:%d\n", err);
                //fprintf(stderr, "wrapper_Receiver: sleeping for 5 seconds:\n");
                //sleep(5);
                //fprintf(stderr, "wrapper_Receiver: a arr address:%lu\n", sg_a_arr[i]);
                //fprintf(stderr, "wrapper_Receiver: a arr with size:%d\n", sg_matrix_meta[i].m * sg_matrix_meta[i].k);
                test_print_arr(sg_a_arr[i], sg_matrix_meta[i].m, sg_matrix_meta[i].k);
                printed = 1;
            }*/
            //MPI_Recv(sg_b_arr[i], sg_matrix_meta[i].k * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * i + DATA_B_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            //sg_notreceived[i] = 0;
            //sem_post(r_lock + i);    
            //fprintf(stderr, "wrapper_Receiver: release rlock:%d\n", i);        
        }
        

        //fprintf(stderr, "wrapper_Receiver: release ilock\n");
        

        r_count = 0;
        flag = 0;
        while (r_count < 2 * s_count){
            //fprintf(stderr, "wrapper_Receiver: in loop\n");
            MPI_Testany(s_count, data_a_status, &i, &flag, MPI_STATUS_IGNORE);
            if (flag == TRUE && i != MPI_UNDEFINED){
                notreceived[i] -= 1;  
                ++r_count;
                if (notreceived[i] == 0){
                    sem_post(r_lock + i);
                }
            }
            MPI_Testany(s_count, data_b_status, &i, &flag, MPI_STATUS_IGNORE);
            if (flag == TRUE && i != MPI_UNDEFINED){
                notreceived[i] -= 1; 
                ++r_count; 
                if (notreceived[i] == 0){
                    sem_post(r_lock + i);
                }
            }
        }
        //MPI_Waitall(s_count, data_a_status, MPI_STATUSES_IGNORE);
        //MPI_Waitall(s_count, data_b_status, MPI_STATUSES_IGNORE);
        //for (i = s_count - 1; i >= 0; --i){
        //    sg_notreceived[i] = -1;
        //}
        
        //fprintf(stderr, "wrapper_Receiver: Received a and b\n");
            
        //flag = s_count;
        //while (flag > 0){   
        for (i = s_count - 1; i >= 0; --i){
            //fprintf(stderr, "wrapper_Receiver: want plock:%d\n", i);
            sem_wait(p_lock + i);
            //fprintf(stderr, "wrapper_Receiver: got plock:%d\n", i);
                //if (sg_notreceived[i] == 0){
            //fprintf(stderr, "wrapper_Receiver: Sending out c_arr[%d]\n", i);
            //fprintf(stderr, "wrapper_Receiver: sending result[%d] of global ID = %d of %d X %d\n", i, s_i[i], sg_matrix_meta[i].m, sg_matrix_meta[i].n);
            MPI_Isend(sg_c_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + RESULT_TAG, MPI_COMM_WORLD, result_status + i);
            //MPI_Ssend(sg_c_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD);
            //sg_notreceived[i] = 1;
                //--flag;
/*                } else{*/
/*                    sem_post(w_lock + i); */
/*                    fprintf(stderr, "wrapper_Receiver: release wlock:%d\n", i);          */
/*                }*/
        }
        //}
        //fprintf(stderr, "wrapper_Receiver: Sent out all c arrs\n");

        MPI_Waitall(s_count, result_status, MPI_STATUSES_IGNORE);
        //fprintf(stderr, "wrapper_Receiver: want ilock\n");
        sem_wait(&i_lock);
        //fprintf(stderr, "wrapper_Receiver: got ilock\n");
        MPI_Irecv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);   
        s_count = sg_count;     
        sem_post(&i_lock);
        //fprintf(stderr, "wrapper_Receiver: release ilock\n");
    }       
    

/*    MPI_Irecv(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);*/
/*    MPI_Wait(&signal_status, MPI_STATUS_IGNORE);*/
    //fprintf(stderr, "wrapper_Receiver: want ilock\n");
    sem_wait(&i_lock);
    //fprintf(stderr, "wrapper_Receiver: got ilock\n");
    sg_count = -1;
    sem_post(&i_lock);
    //fprintf(stderr, "wrapper_Receiver: release ilock, exit\n");
    for (i = 0; i < NUM_OF_WTHREADS; i++){
        sem_post(&t_lock);
        //fprintf(stderr, "wrapper_Receiver: release tlock\n");  
    }
    
    pthread_exit(NULL); 


}

void *wrapper_Worker(void *data){
    int core_id = (long) data;
    //stick_to_core(core_id);
    int i;
    //int total_count = 0;
    //int s_i[WGRAPE_SIZE] = {0};
    while (TRUE){
        //fprintf(stderr, "wrapper_Worker: want tlock\n");
        sem_wait(&t_lock);
        //fprintf(stderr, "wrapper_Worker: got tlock\n");
        //fprintf(stderr, "wrapper_Worker: want ilock\n");
        sem_wait(&i_lock);
        //fprintf(stderr, "wrapper_Worker: got ilock\n");
        if (sg_count > 0){
            --sg_count;
            i = sg_count;
            sem_post(&i_lock);
            //fprintf(stderr, "wrapper_Worker: release ilock, want rlock:%d\n", i);  
            sem_wait(r_lock + i);
            //fprintf(stderr, "wrapper_Worker: got rlock:%d\n", i);
            NPdgemm(sg_matrix_meta[i].tra_a, sg_matrix_meta[i].tra_b,
                sg_matrix_meta[i].m, sg_matrix_meta[i].n, sg_matrix_meta[i].k,
                sg_matrix_meta[i].lda, sg_matrix_meta[i].ldb, sg_matrix_meta[i].ldc,
                sg_matrix_meta[i].offseta, sg_matrix_meta[i].offsetb,                      
                sg_matrix_meta[i].offsetc, sg_a_arr[i], sg_b_arr[i], sg_c_arr[i],
                sg_matrix_meta[i].alpha, sg_matrix_meta[i].beta); 
            //++total_count;
            //fprintf(stderr, "wrapper_Worker: after NPdgemm, sg_c_arr[%d][0]=%lf\n", i, sg_c_arr[i][0]);
            sem_post(p_lock + i); 
            //fprintf(stderr, "wrapper_Worker: release plock:%d\n", i);
        } else if (sg_count < 0){
            sem_post(&i_lock);
            //fprintf(stderr, "wrapper_Worker: release ilock and exit\n");
            break;
        } else {
            sem_post(&i_lock);
            //fprintf(stderr, "wrapper_Worker: release ilock, no work, want tlock\n");
        }
         
        
    }
    //fprintf(stderr, "Worker thread was distributed %d tasks\n", total_count);
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
        //sleep(0);
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
	    fprintf(stderr, "mpi_lNPdgemm: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
	    if (NUM_OF_WTHREADS > 0){
            for (i = 0; i < WORLD_SIZE - 1; ++i){
                pthread_join(driving_threads[i], NULL);
            }
        }        
 	    
        free(driving_threads);
        free(dest);
        
    }else{
        //Worker:  
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
    fprintf(stderr, "mpi_lNPdgemm: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
        
    //MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);  
    //pthread_exit(NULL); 
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
    fprintf(stderr, "pth_lNPdgemm: %f seconds taken.\n", omp_get_wtime() - stime);
}
