#include "mpi_helper.h"   
#include "omp_helper.h" 
#include "blas_helper.h"
static int sg_i = 0;
static int sg_s_i[WGRAPE_SIZE];
static int sg_count = 0;
static sem_t i_lock;
static sem_t t_lock;
static sem_t r_lock[WGRAPE_SIZE];
static sem_t p_lock[WGRAPE_SIZE];

static int sg_len_pl_tdi, sg_len_s_f, sg_len_tdim, sg_len_pll, sg_mfno;
static char *sg_tra_tdi, *sg_tra_sm, *sg_tra_tbar, *sg_tra_result;
static int *sg_pairlist_local, *sg_indices, *sg_m_tdi, *sg_n_tdi, *sg_m_tbar, *sg_n_tbar, *sg_m_sm, *sg_n_sm, *sg_m_result, *sg_n_result, *sg_t_dim;
static double **sg_T_di, **sg_T_bar, **sg_S_matrix, **sg_result;

//static NP_MATRIX **sg_result_matrix;

extern MPI_Datatype MPI_MATRIX_META;
extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];


void *wrapper_genpresidue_Local(void *data){
    int core_id = (long) data;
    //stick_to_core(core_id);
    int ind, index, ipair, i, j;
    int s_i[CGRAPE_SIZE] = {0};
    int s_count = 0;
    NP_MATRIX tdi_wrapper, bar_wrapper, result_wrapper;
    NP_MATRIX *temp_supermatrix, *temp_ddot_result, *temp_final_result, *ptr2result;
    temp_supermatrix = NULL;
    temp_ddot_result = NULL;
	temp_final_result = NULL;

    while (TRUE){
        sem_wait(&i_lock);
        while (s_count < CGRAPE_SIZE){                
            if (sg_i >= sg_len_pll){
                break;
            } else {
                s_i[s_count++] = sg_i++;
            }
        }
        sem_post(&i_lock);
        while (s_count > 0){
            ind = s_i[--s_count];
			
            ///Start actual computation
			ipair = sg_pairlist_local[ind];
            i = ipair / sg_mfno;
            j = ipair % sg_mfno;
            index = sg_indices[ind];
            elogf("original index=%d\n", ind);
            
			wrap_np_matrix(&tdi_wrapper, sg_T_di[index], sg_m_tdi[index], sg_n_tdi[index], sg_tra_tdi[index]);
			generate_supermatrix(&temp_supermatrix, i, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
			
            ddot_np_matrix(&temp_ddot_result, &tdi_wrapper, temp_supermatrix);
			wrap_np_matrix(&result_wrapper, sg_result[ind], sg_m_result[ind], sg_n_result[ind], sg_tra_result[ind]);
            wrap_np_matrix(&bar_wrapper, sg_T_bar[index], sg_m_tbar[index], sg_n_tbar[index], sg_tra_tbar[index]);
			transpose_np_matrix(&bar_wrapper);
			ptr2result = &result_wrapper;
			ddot_np_matrix(&ptr2result, temp_ddot_result, &bar_wrapper);
			
			ddot_np_matrix(&temp_ddot_result, &bar_wrapper, temp_supermatrix);
			ddot_np_matrix(&temp_final_result, temp_ddot_result, &tdi_wrapper);
			ptr2result = &result_wrapper;
			axpy_np_matrix(&ptr2result, temp_final_result, 1, &result_wrapper);
			if (i != j){
				scal_np_matrix(&ptr2result, &result_wrapper, 2);
			}
            ///End actual computation
            //flogf(stderr, "%s: %s: %s\n", __FILE__, __func__, "end of actual computation");
        }       
        
        if (sg_i >= sg_len_pl_tdi){
            delete_np_matrix(&temp_ddot_result);
            delete_np_matrix(&temp_supermatrix);
            delete_np_matrix(&temp_final_result);
            pthread_exit(NULL); 
        }        
    }
} 

void *wrapper_genpresidue_Driver(void *data){//finished
    int total_count = 0;
    int i, temp_sc, s_count;
    int dest = *(int *)data;
    MPI_Request result_status[WGRAPE_SIZE];
    int s_i[WGRAPE_SIZE] = {0};
    while (TRUE){
        s_count = 0;
        sem_wait(&i_lock);
        
        while (s_count < WGRAPE_SIZE){
            if (sg_i < sg_len_pl_tdi){
                s_i[s_count] = sg_i;
                ++s_count;
                ++sg_i;
                ++total_count;
            } else{
                break;
            }
        }
        if (sg_i >= sg_len_pl_tdi && s_count <= 0){
            sem_post(&i_lock);
            flogf(stderr, "mpi_lgenpresidue: dest=%d was distributed %d tasks\n", dest, total_count);
            MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
            pthread_exit(NULL); 
        } 
        sem_post(&i_lock);
        MPI_Ssend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
        MPI_Ssend(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD);
        temp_sc = s_count;
        
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            MPI_Irecv(sg_result[i], sg_m_result[i] * sg_n_result[i], MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + RESULT_TAG, MPI_COMM_WORLD, result_status + s_count);
        }
        MPI_Waitall(temp_sc, result_status, MPI_STATUSES_IGNORE);
    }
}  

void *wrapper_genpresidue_Receiver(void *data){//finished
    int locali, ind;
    int s_count = 0;
    int dest = 0; // to root
	//int s_i[WGRAPE_SIZE];
    MPI_Request meta_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE];
    
    sem_wait(&i_lock);
    MPI_Recv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    s_count = sg_count;
    sem_post(&i_lock);
	
    while (s_count > 0){
		
		MPI_Recv(sg_s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
        for (locali = s_count - 1; locali >= 0; --locali){
            //ind = sg_s_i[locali];
            //sg_result[locali] = realloc(sg_result[locali], sizeof(double) * sg_m_result[ind] * sg_n_result[ind]);  
            sem_post(&t_lock);  
            sem_post(r_lock + locali); 
        }

        for (locali = s_count - 1; locali >= 0; --locali){
			ind = sg_s_i[locali];
            sem_wait(p_lock + locali);
            MPI_Isend(sg_result[ind], sg_m_result[ind] * sg_n_result[ind], MPI_DOUBLE, dest, NUM_OF_TAGS * locali + RESULT_TAG, MPI_COMM_WORLD, result_status + locali);
        }
        MPI_Waitall(s_count, result_status, MPI_STATUSES_IGNORE);
        sem_wait(&i_lock);
        MPI_Recv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   
        s_count = sg_count;     
        sem_post(&i_lock);
    }       
    sem_wait(&i_lock);
    sg_count = -1;//-1 means no more for this call, 0 means temporarily no more
    sem_post(&i_lock);
    for (locali = 0; locali < NUM_OF_WTHREADS; locali++){
        sem_post(&t_lock);
    }
    pthread_exit(NULL); 
}


void *wrapper_genpresidue_Worker(void *data){//finished
    int core_id = (long) data;
    //stick_to_core(core_id);
    int ind, locali, ipair, i, j, index;
    NP_MATRIX tdi_wrapper, bar_wrapper, result_wrapper;
    NP_MATRIX *temp_supermatrix, *temp_ddot_result, *temp_final_result, *ptr2result;
    temp_supermatrix = NULL;
    temp_ddot_result = NULL;
	temp_final_result = NULL;
	
    while (TRUE){
        sem_wait(&t_lock);
        sem_wait(&i_lock);
        if (sg_count > 0){
            --sg_count;
			locali = sg_count;
            ind = sg_s_i[locali];
            sem_post(&i_lock);
            sem_wait(r_lock + locali);
			
			
            ///Start actual computation
			ipair = sg_pairlist_local[ind];
            i = ipair / sg_mfno;
            j = ipair % sg_mfno;
            index = sg_indices[ind];
            elogf("original index=%d\n", ind);
            
			wrap_np_matrix(&tdi_wrapper, sg_T_di[index], sg_m_tdi[index], sg_n_tdi[index], sg_tra_tdi[index]);
			generate_supermatrix(&temp_supermatrix, i, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
			
            ddot_np_matrix(&temp_ddot_result, &tdi_wrapper, temp_supermatrix);
			wrap_np_matrix(&result_wrapper, sg_result[ind], sg_m_result[ind], sg_n_result[ind], sg_tra_result[ind]);
            wrap_np_matrix(&bar_wrapper, sg_T_bar[index], sg_m_tbar[index], sg_n_tbar[index], sg_tra_tbar[index]);
			transpose_np_matrix(&bar_wrapper);
			ptr2result = &result_wrapper;
			ddot_np_matrix(&ptr2result, temp_ddot_result, &bar_wrapper);
			
			ddot_np_matrix(&temp_ddot_result, &bar_wrapper, temp_supermatrix);
			ddot_np_matrix(&temp_final_result, temp_ddot_result, &tdi_wrapper);
			ptr2result = &result_wrapper;
			axpy_np_matrix(&ptr2result, temp_final_result, 1, &result_wrapper);
			if (i != j){
				scal_np_matrix(&ptr2result, &result_wrapper, 2);
			}
            ///End actual computation
			

            sem_post(p_lock + locali); 
        } else if (sg_count < 0){
            sem_post(&i_lock);

            delete_np_matrix(&temp_ddot_result);
            delete_np_matrix(&temp_supermatrix);
            delete_np_matrix(&temp_final_result);
            break;
        } else {
            sem_post(&i_lock);
        }
    }
    pthread_exit(NULL); 
}

void mpi_lgenpresidue(int len_pl_tdi, 
                    double** T_di, int* m_tdi, int* n_tdi, char* tra_tdi, 
                    double** T_bar, int* m_tbar, int* n_tbar, char* tra_tbar,
                    int len_s_f,
                    double** S_matrix, int *m_sm, int *n_sm, char* tra_sm,
                    int len_tdim, int* t_dim,
                    int len_pll, int* pairlist_local, int *indices,
                    double** result, int* m_result, int* n_result, char* tra_result,
					int mfno){
    double stime = omp_get_wtime();  
	int signal = SIGNAL_LGENPRESIDUE;  
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }    
	
	//Bcast everything
	MPI_Bcast(&len_pl_tdi  , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&len_s_f     , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&len_tdim    , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&len_pll    , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&mfno       , 1            , MPI_INT , 0, MPI_COMM_WORLD);

	if (WORLD_RANK != 0){
		pairlist_local = malloc(sizeof(int) * len_pll);
		m_tdi = malloc(sizeof(int) * len_pl_tdi);
		n_tdi = malloc(sizeof(int) * len_pl_tdi);
		tra_tdi = malloc(sizeof(char) * len_pl_tdi);
		m_tbar = malloc(sizeof(int) * len_pl_tdi);
		n_tbar = malloc(sizeof(int) * len_pl_tdi);
		tra_tbar = malloc(sizeof(char) * len_pl_tdi);		
		m_sm = malloc(sizeof(int) * len_s_f);
		n_sm = malloc(sizeof(int) * len_s_f);
		tra_sm = malloc(sizeof(char) * len_s_f);
		indices = malloc(sizeof(int) * len_pll);
		m_result = malloc(sizeof(int) * len_pll);
		n_result = malloc(sizeof(int) * len_pll);
		tra_result = malloc(sizeof(char) * len_pll);		
		t_dim = malloc(sizeof(int) * len_tdim);
		T_di = malloc(sizeof(double*) * len_pl_tdi);
		T_bar = malloc(sizeof(double*) * len_pl_tdi);
		result = malloc(sizeof(double*) * len_pll);
		S_matrix = malloc(sizeof(double*) * len_s_f);
	}
	
	
	
	MPI_Bcast(m_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_tdi      , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_tbar       , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_tbar       , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_tbar     , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(m_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_sm       , len_s_f   , MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(pairlist_local,len_pll   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(indices      , len_pll   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(m_result     , len_pll   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_result     , len_pll   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_result   , len_pll   , MPI_CHAR, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(t_dim        , len_tdim  , MPI_INT , 0, MPI_COMM_WORLD);

	
	flogf(stderr, "mpi_lgenpresidue: %f seconds for all Bcasts to finish on processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
	
	sg_len_pl_tdi = len_pl_tdi;
	sg_pairlist_local = pairlist_local;       ///ASSUME PAIRLIST IS SORTED!!!
	sg_T_di = T_di;//sendlatter
	sg_m_tdi = m_tdi;
	sg_n_tdi = n_tdi;
	sg_tra_tdi = tra_tdi;
	sg_T_bar = T_bar;//sendlatter
	sg_m_tbar = m_tbar;
	sg_n_tbar = n_tbar;
	sg_tra_tbar = tra_tbar;	
	
	
	sg_len_s_f = len_s_f;
	sg_S_matrix = S_matrix;//sendlatter
	sg_m_sm = m_sm;
	sg_n_sm = n_sm;
	sg_tra_sm = tra_sm;
	
	sg_len_pll = len_pll;
	sg_m_result = m_result;
	sg_n_result = n_result;
	sg_tra_result = tra_result;	
	sg_indices = indices;
	sg_result = result;
	
	sg_len_tdim = len_tdim;
	sg_t_dim =t_dim;
	
	sg_mfno = mfno;
	
	sg_i = 0;   
	
	
	MPI_Request *tdi_status, *sm_status, *tbar_status;
	tdi_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	tbar_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	sm_status = malloc(sizeof(MPI_Request) * len_s_f);

    int i, *dest;
    sem_init(&i_lock, 0, 1);
	
	//Root:
	if (WORLD_RANK == 0){ 

		flogf(stderr, "mpi_lgenpresidue: total %d tasks\n", sg_len_pl_tdi);

		dest = malloc(sizeof(int) * (WORLD_SIZE - 1));
		pthread_t *driving_threads = malloc(sizeof(pthread_t) * NUM_OF_WTHREADS * (WORLD_SIZE - 1));
		
		pthread_t computing_threads[NUM_OF_CTHREADS];    
			 
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_create(computing_threads + i, NULL, wrapper_genpresidue_Local, (void*)i);
		}
		
		for (i = 0; i < WORLD_SIZE - 1; ++i){
			dest[i] = i + 1;
		}		
		
		
		
		for (i = 0; i < len_pl_tdi; ++i){
			MPI_Ibcast(T_di[i], m_tdi[i] * n_tdi[i], MPI_DOUBLE, 0, MPI_COMM_WORLD, tdi_status + i);
			MPI_Ibcast(T_bar[i], m_tbar[i] * n_tbar[i], MPI_DOUBLE, 0, MPI_COMM_WORLD, tbar_status + i);
		}
		

		
		for (i = 0; i < len_s_f; ++i){
			MPI_Ibcast(S_matrix[i], m_sm[i] * n_sm[i], MPI_DOUBLE, 0, MPI_COMM_WORLD, sm_status + i);
		}
		
		if (WORLD_SIZE > 1){
			MPI_Waitall(len_pl_tdi, tdi_status, MPI_STATUSES_IGNORE);
			MPI_Waitall(len_pl_tdi, tbar_status, MPI_STATUSES_IGNORE);			
			MPI_Waitall(len_s_f, sm_status, MPI_STATUSES_IGNORE);
		}
		flogf(stderr, "%s: %s: %s\n", __FILE__, __func__, "sent T_di, T_bar, S_matrix");		
		
		
		if (NUM_OF_WTHREADS > 0){
			for (i = 0; i < WORLD_SIZE - 1; ++i){
				pthread_create(driving_threads + i, NULL, wrapper_genpresidue_Driver, dest + i);
			}
		}
		
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_join(computing_threads[i], NULL);
		}
		flogf(stderr, "mpi_lgenpresidue: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
		
		if (NUM_OF_WTHREADS > 0){
			for (i = 0; i < WORLD_SIZE - 1; ++i){
				pthread_join(driving_threads[i], NULL);
			}
		}   
		
		free(driving_threads);
		free(dest);
	} else{
		//worker
		
		int temp_size;
		for (i = 0; i < len_pl_tdi; ++i){
			temp_size = sg_m_tdi[i] * sg_n_tdi[i];
			sg_T_di[i] = malloc(sizeof(double) * temp_size);
			MPI_Ibcast(sg_T_di[i], temp_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, tdi_status + i);
			
			temp_size = sg_m_tbar[i] * sg_n_tbar[i];
			sg_T_bar[i] = malloc(sizeof(double) * temp_size);
			MPI_Ibcast(sg_T_bar[i], temp_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, tbar_status + i);			
		}
		
		for (i = 0; i < len_s_f; ++i){
			temp_size = sg_m_sm[i] * sg_n_sm[i];
			sg_S_matrix[i] = malloc(sizeof(double) * temp_size);
			MPI_Ibcast(sg_S_matrix[i], temp_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, sm_status + i);			
		}
		
		for (i = 0; i < len_pll; ++i){
			temp_size = sg_m_result[i] * sg_n_result[i];
			sg_result[i] = malloc(sizeof(double) * temp_size);
		}
		
		MPI_Waitall(len_pl_tdi, tdi_status, MPI_STATUSES_IGNORE);
		MPI_Waitall(len_pl_tdi, tbar_status, MPI_STATUSES_IGNORE);
		MPI_Waitall(len_s_f, sm_status, MPI_STATUSES_IGNORE);
		
        sem_init(&t_lock, 0, 0); //task lock : how many unfinished task in current comm
		
        for (i = 0; i < WGRAPE_SIZE; ++i){
            sem_init(r_lock + i, 0, 0); //resource lock : if current task resource is ready
            sem_init(p_lock + i, 0, 0); //product lock : if current product is ready
        }
        
        pthread_t working_threads[NUM_OF_WTHREADS];
        pthread_t receiving_thread;
        pthread_create(&receiving_thread, NULL, wrapper_genpresidue_Receiver, NULL);
        for (i = 0; i < NUM_OF_WTHREADS; ++i){
            pthread_create(working_threads + i, NULL, wrapper_genpresidue_Worker, (void*) i);
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
		
		for (i = 0; i < len_pl_tdi; ++i){
			free(sg_T_di[i]);
			free(sg_T_bar[i]);
		}
		
		for (i = 0; i < len_s_f; ++i){
			free(sg_S_matrix[i]);
		}
		
		for (i = 0; i < len_pll; ++i){
			free(sg_result[i]);
		}
		free(pairlist_local);
		free(m_tdi);
		free(n_tdi);
		free(tra_tdi);
		free(m_tbar);
		free(n_tbar);
		free(tra_tbar);
		free(m_sm);
		free(n_sm);
		free(tra_sm);
		free(indices);
		free(m_result);
		free(n_result);
		free(tra_result);
		free(t_dim);
		free(T_di);
		free(T_bar);
		free(result);
		free(S_matrix);

	}
	free(tdi_status);
	free(tbar_status);
	free(sm_status);
	
    sem_destroy(&i_lock);
	flogf(stderr, "mpi_lgenpresidue: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
}

/*
void pth_lgpresidue(int len_pl_tdi, 
                   int* completed, int* pairlist, 
                   double** T_di, int* m_tdi, int* n_tdi, char* tra_tdi, 
                   double** result, int* m_result, int* n_result, char* tra_result,
                   int len_s_f,
                   double** S_matrix, char* tra_sm,
                   double** F_matrix, char* tra_fm,
                   double* loc_fork, int m_lf, int n_lf, char tra_lf,
                   int len_tdim, int* t_dim,
                   int mfno, int nonredundant){
    double stime = omp_get_wtime();  
    int i;
    sem_init(&i_lock, 0, 1);
    sg_len_pl_tdi = len_pl_tdi;
    sg_completed = completed;
    sg_pairlist = pairlist;       ///ASSUME PAIRLIST IS SORTED!!!
    sg_T_di = T_di;
    sg_m_tdi = m_tdi;
    sg_n_tdi = n_tdi;
    sg_tra_tdi = tra_tdi;
    sg_result = result;
    sg_m_result = m_result;
    sg_n_result = n_result;
    sg_tra_result = tra_result;
    sg_len_s_f = len_s_f;
    sg_S_matrix = S_matrix;
    sg_tra_sm = tra_sm;
    sg_F_matrix = F_matrix;
    sg_tra_fm = tra_fm;
    wrap_np_matrix(&sg_np_lf, loc_fork, m_lf, n_lf, tra_lf);
    
    sg_len_tdim = len_tdim;
    sg_t_dim =t_dim;
    sg_mfno = mfno;
    sg_nonredundant = nonredundant;
    sg_i = 0;   
    pthread_t computing_threads[NUM_OF_CTHREADS];    
    
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_create(computing_threads + i, NULL, wrapper_gpresidue_Local, NULL);
    }
    
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_join(computing_threads[i], NULL);
    }
 	    
    sem_destroy(&i_lock);
    flogf(stderr, "pth_lgpresidue: %f seconds taken.\n", omp_get_wtime() - stime);
}
*/
