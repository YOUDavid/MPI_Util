#include "mpi_helper.h"   
#include "omp_helper.h" 
#define DEBUG 0

extern MPI_Datatype MPI_MATRIX_META;
extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];


static int sg_current_work_index = 0;
static int sg_current_work_count = 0;
//static int sg_notreceived[WGRAPE_SIZE];
static sem_t counter_lock, thread_lock, resource_lock[WGRAPE_SIZE], product_lock[WGRAPE_SIZE];

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

enum TYPE{
    char,
    short,
    int, float,
    long, double,
    long long, long double
};

struct Workpack{
    TYPE


};




extern double *TEMP_ARR;
extern double **A_ARR, **B_ARR, **C_ARR;
static MATRIX_META sg_matrix_meta[WGRAPE_SIZE];



   
void *local_worker(void *data){
    int core_id = (long) data;
    //stick_to_core(core_id);
    int i;
    int s_i[CGRAPE_SIZE] = {0};
    int s_count = 0;
    while (TRUE){
        sem_wait(&counter_lock);
        while (s_count < CGRAPE_SIZE){
            
            while (sg_current_work_index < sg_matrix_num && sg_completed[sg_current_work_index] == 1){
                ++sg_current_work_index;
            }
                
            if (sg_current_work_index >= sg_matrix_num){
                sem_post(&counter_lock);
                break;
            } else {
                s_i[s_count++] = sg_current_work_index++;
            }
        }
        sem_post(&counter_lock);
        while (s_count > 0){
            i = s_i[--s_count];
            NPdgemm(sg_tra_a[i], sg_tra_b[i], sg_m_arr[i], sg_n_arr[i], sg_k_arr[i], sg_lda[i], 
                sg_ldb[i], sg_ldc[i], sg_offseta[i], sg_offsetb[i], sg_offsetc[i], sg_a_arr[i],
                sg_b_arr[i], sg_c_arr[i], sg_alpha[i], sg_beta[i]);

        }       
        
        if (sg_current_work_index >= sg_matrix_num){
            pthread_exit(NULL); 
        }        
    }
} 

void *driver(void *data){
    int total_count = 0;
    int i, temp_sc, s_count;
    int dest = *(int *)data;
    MPI_Request meta_status[WGRAPE_SIZE], data_a_status[WGRAPE_SIZE], data_b_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE], signal_status;
	MATRIX_META matrix_meta[WGRAPE_SIZE];
    int s_i[WGRAPE_SIZE] = {0};
    
    while (TRUE){
        s_count = 0;
        //fprintf(stderr, "wrapper_Driver: want ilock\n");
        sem_wait(&counter_lock);
        //fprintf(stderr, "wrapper_Driver: got ilock\n");
        
        while (s_count < WGRAPE_SIZE){
            
            while (sg_current_work_index < sg_matrix_num && sg_completed[sg_current_work_index] == 1){
                ++sg_current_work_index;
            }
                
            if (sg_current_work_index < sg_matrix_num){
                s_i[s_count] = sg_current_work_index;
                ++s_count;
                ++sg_current_work_index;
                ++total_count;
            } else{
                break;
            }
        }
        if (sg_current_work_index >= sg_matrix_num && s_count <= 0){
            sem_post(&counter_lock);
            //fprintf(stderr, "wrapper_Driver: release ilock\n");
            fprintf(stderr, "wrapper_Driver: dest=%d was distributed %d tasks\n", dest, total_count);
            //MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
            MPI_Isend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
            MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
            pthread_exit(NULL); 
        } 
        sem_post(&counter_lock);
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

void *receiver(void *data){
    int i, flag, r_count;
    int s_count = 0;
    int dest = 0; // to root
    int notreceived[WGRAPE_SIZE];
	int s_i[WGRAPE_SIZE];
    MPI_Request meta_status[WGRAPE_SIZE], data_a_status[WGRAPE_SIZE], data_b_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE], signal_status;
    
    sem_wait(&counter_lock);
    MPI_Irecv(&sg_current_work_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
    MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
    s_count = sg_current_work_count;
	
    sem_post(&counter_lock);
    while (s_count > 0){
		
		MPI_Irecv(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
		MPI_Wait(&signal_status, MPI_STATUS_IGNORE);	
		
        for (i = s_count - 1; i >= 0; --i){
            MPI_Irecv(sg_matrix_meta + i, 1, MPI_MATRIX_META, dest, NUM_OF_TAGS * s_i[i] + META_TAG, MPI_COMM_WORLD, meta_status + i);
			
            sem_post(&thread_lock);    
        }
        for (i = s_count - 1; i >= 0; --i){
            MPI_Wait(meta_status + i, MPI_STATUS_IGNORE);
            notreceived[i] = 2;
            MPI_Irecv(sg_a_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].k, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + DATA_A_TAG, MPI_COMM_WORLD, data_a_status + i);              
            MPI_Irecv(sg_b_arr[i], sg_matrix_meta[i].k * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + DATA_B_TAG, MPI_COMM_WORLD, data_b_status + i);   
        }
        r_count = 0;
        flag = 0;
        while (r_count < 2 * s_count){
            MPI_Testany(s_count, data_a_status, &i, &flag, MPI_STATUS_IGNORE);
            if (flag == TRUE && i != MPI_UNDEFINED){
                notreceived[i] -= 1;  
                ++r_count;
                if (notreceived[i] == 0){
                    sem_post(resource_lock + i);
                }
            }
            MPI_Testany(s_count, data_b_status, &i, &flag, MPI_STATUS_IGNORE);
            if (flag == TRUE && i != MPI_UNDEFINED){
                notreceived[i] -= 1; 
                ++r_count; 
                if (notreceived[i] == 0){
                    sem_post(resource_lock + i);
                }
            }
        }
        for (i = s_count - 1; i >= 0; --i){
            sem_wait(product_lock + i);
            MPI_Isend(sg_c_arr[i], sg_matrix_meta[i].m * sg_matrix_meta[i].n, MPI_DOUBLE, dest, NUM_OF_TAGS * s_i[i] + RESULT_TAG, MPI_COMM_WORLD, result_status + i);
        }

        MPI_Waitall(s_count, result_status, MPI_STATUSES_IGNORE);
        sem_wait(&counter_lock);
        MPI_Irecv(&sg_current_work_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);   
        s_count = sg_current_work_count;     
        sem_post(&counter_lock);
    }       
    sem_wait(&counter_lock);
    sg_current_work_count = -1;
    sem_post(&counter_lock);
    for (i = 0; i < NUM_OF_WTHREADS; i++){
        sem_post(&thread_lock);
    }
    pthread_exit(NULL); 
}

void *remote_worker(void *data){
    int core_id = (long) data;
    int i;
    while (TRUE){
        sem_wait(&thread_lock);
        sem_wait(&counter_lock);
        if (sg_current_work_count > 0){
            --sg_current_work_count;
            i = sg_current_work_count;
            sem_post(&counter_lock);
            sem_wait(resource_lock + i);
            NPdgemm(sg_matrix_meta[i].tra_a, sg_matrix_meta[i].tra_b,
                sg_matrix_meta[i].m, sg_matrix_meta[i].n, sg_matrix_meta[i].k,
                sg_matrix_meta[i].lda, sg_matrix_meta[i].ldb, sg_matrix_meta[i].ldc,
                sg_matrix_meta[i].offseta, sg_matrix_meta[i].offsetb,                      
                sg_matrix_meta[i].offsetc, sg_a_arr[i], sg_b_arr[i], sg_c_arr[i],
                sg_matrix_meta[i].alpha, sg_matrix_meta[i].beta); 
            sem_post(product_lock + i); 
        } else if (sg_current_work_count < 0){
            sem_post(&counter_lock);
            break;
        } else {
            sem_post(&counter_lock);
        }
    }
    pthread_exit(NULL); 
    

}

void mpi_template(){    
    double stime = omp_get_wtime();                 
    int signal = SIGNAL_LNPDGEMM;     
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }    
    int i, *dest;
    sem_init(&counter_lock, 0, 1);
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
        sg_current_work_index = 0;   

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
        sem_init(&thread_lock, 0, 0); //task lock : how many unfinished task in current comm
        sg_current_work_index = 0;
        sg_temp_arr = TEMP_ARR;
        sg_a_arr = A_ARR;
        sg_b_arr = B_ARR;
        sg_c_arr = C_ARR;
        for (i = 0; i < WGRAPE_SIZE; ++i){
            sem_init(resource_lock + i, 0, 0); //resource lock : if current task resource is ready
            sem_init(product_lock + i, 0, 0); //product lock : if current product is ready
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
            sem_destroy(resource_lock + i);
            sem_destroy(product_lock + i);
        }
        sem_destroy(&thread_lock);

        

              
    } 
    sem_destroy(&counter_lock);
    fprintf(stderr, "mpi_lNPdgemm: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
        
    //MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);  
    //pthread_exit(NULL); 
}

void pth_template(){    
    double stime = omp_get_wtime();                 
    int i;
    sem_init(&counter_lock, 0, 1);
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
    sg_current_work_index = 0;   
    pthread_t computing_threads[NUM_OF_CTHREADS];            
        
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_create(computing_threads + i, NULL, wrapper_NPdgemm, NULL);
    }
    
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_join(computing_threads[i], NULL);
    }
 	    
    sem_destroy(&counter_lock);
    fprintf(stderr, "pth_lNPdgemm: %f seconds taken.\n", omp_get_wtime() - stime);
}
