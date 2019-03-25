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

static int sg_len_pl_tdi, sg_len_s_f, sg_len_tdim, sg_mfno, sg_nonredundant, sg_len_pll;
static char *sg_tra_tdi, *sg_tra_sm, *sg_tra_tbar;
static int *sg_pairlist, *sg_pairlist_local, *sg_m_tdi, *sg_n_tdi, *sg_m_tbar, *sg_n_tbar, *sg_m_sm, *sg_n_sm, *sg_t_dim;
static double **sg_T_di, **sg_T_bar, **sg_S_matrix, *sg_result;

//static NP_MATRIX **sg_result_matrix;

extern MPI_Datatype MPI_MATRIX_META;
extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];


void *wrapper_gpresidue_Local(void *data){
    int core_id = (long) data;
    //stick_to_core(core_id);
    int ind, ipair, i, j, k;
    int s_i[CGRAPE_SIZE] = {0};
	double r1, r2;
    int s_count = 0;
    NP_MATRIX temp_wrapper;
    NP_MATRIX *temp_smikijsm, *temp_smijkjsm, *temp_smijiksm, *temp_smkjijsm, *temp_flip_result, *temp_ddot_result, *A, *B, *C, *D;
    temp_smikijsm = NULL;
    temp_smijkjsm = NULL;
    temp_smijiksm = NULL;
    temp_smkjijsm = NULL;
    temp_flip_result = NULL;
    temp_ddot_result = NULL;
    A = NULL;
    B = NULL;
    C = NULL;
    D = NULL;

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
            k = ipair / sg_mfno;
            j = ipair % sg_mfno;
            r1 = 0.0;
            elogf("original index=%d\n", ind);
            
            for (i = 0; i < sg_mfno; ++i){
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0) ||
                    (i < j && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + j) < 0) ||
                    (j < i && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + i) < 0)){
                    continue;
                }
				generate_supermatrix(&temp_smijiksm, i, j, i, k, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                generate_supermatrix(&temp_smikijsm, i, k, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                
                if (sg_nonredundant == 1 && i > k){
                    int idxki = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxki], sg_m_tdi[idxki], sg_n_tdi[idxki], sg_tra_tdi[idxki]);
                    
                    flip_ij(&temp_flip_result, k, i, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&temp_ddot_result, temp_smijiksm, temp_flip_result);
                    
                    
                } else{
                    int idxik = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxik], sg_m_tdi[idxik], sg_n_tdi[idxik], sg_tra_tdi[idxik]);
                    ddot_np_matrix(&temp_ddot_result, temp_smijiksm, &temp_wrapper);
                }
				ddot_np_matrix(&A, temp_ddot_result, temp_smikijsm);
				
				
				if (sg_nonredundant == 1 && i > j){
                    int idxji = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxji], sg_m_tbar[idxji], sg_n_tbar[idxji], sg_tra_tbar[idxji]); 
					flip_ij(&temp_flip_result, j, i, &temp_wrapper, sg_t_dim);
					temp_flip_result->m *= temp_flip_result->n;
					temp_flip_result->n = 1;
					A->n *= A->m;
					A->m = 1;
                    ddot_np_matrix(&C, A, temp_flip_result);
                    
                } else{
                    int idxij = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxij], sg_m_tbar[idxij] * sg_n_tbar[idxij], 1, sg_tra_tbar[idxij]); 
					A->n *= A->m;
					A->m = 1;
					ddot_np_matrix(&C, A, &temp_wrapper);
                }
				r1 += index_np_matrix(C, 0, 0);
			}
			
			
			
			i = ipair / sg_mfno;
            k = ipair % sg_mfno;
			r2 = 0.0;
            for (j = 0; j < sg_mfno; ++j){
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0) ||
                    (i < j && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + j) < 0) ||
                    (j < i && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + i) < 0)){
                    continue;
                }
				generate_supermatrix(&temp_smijkjsm, i, j, k, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                generate_supermatrix(&temp_smkjijsm, k, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                
                if (sg_nonredundant == 1 && k > j){
                    int idxjk = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxjk], sg_m_tdi[idxjk], sg_n_tdi[idxjk], sg_tra_tdi[idxjk]);
                    
                    flip_ij(&temp_flip_result, j, k, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&temp_ddot_result, temp_smijkjsm, temp_flip_result);
                    
                    
                } else{
                    int idxkj = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxkj], sg_m_tdi[idxkj], sg_n_tdi[idxkj], sg_tra_tdi[idxkj]);
                    ddot_np_matrix(&temp_ddot_result, temp_smijkjsm, &temp_wrapper);
                }
				ddot_np_matrix(&B, temp_ddot_result, temp_smkjijsm);
				
				
				if (sg_nonredundant == 1 && i > j){
                    int idxji = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxji], sg_m_tbar[idxji], sg_n_tbar[idxji], sg_tra_tbar[idxji]); 
					flip_ij(&temp_flip_result, j, i, &temp_wrapper, sg_t_dim);
					temp_flip_result->m *= temp_flip_result->n;
					temp_flip_result->n = 1;
					B->n *= B->m;
					B->m = 1;
                    ddot_np_matrix(&D, B, temp_flip_result);
                    
                } else{
                    int idxij = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxij], sg_m_tbar[idxij] * sg_n_tbar[idxij], 1, sg_tra_tbar[idxij]); 
					B->n *= B->m;
					B->m = 1;
					ddot_np_matrix(&D, B, &temp_wrapper);
                }
				r2 += index_np_matrix(D, 0, 0);
			}
			//flogf(stderr, "sg_result[%d] = %lf;", ind, r1 + r2);
			sg_result[ind] = r1 + r2;
            ///End actual computation
            //flogf(stderr, "%s: %s: %s\n", __FILE__, __func__, "end of actual computation");
        }       
        
        if (sg_i >= sg_len_pl_tdi){
            delete_np_matrix(&temp_smikijsm);
            delete_np_matrix(&temp_smijkjsm);
            delete_np_matrix(&temp_smijiksm);
            delete_np_matrix(&temp_smkjijsm);
            delete_np_matrix(&temp_ddot_result);
            delete_np_matrix(&temp_flip_result);
            delete_np_matrix(&A);
            delete_np_matrix(&B);
            delete_np_matrix(&C);
            delete_np_matrix(&D);
            pthread_exit(NULL); 
        }        
    }
} 

void *wrapper_gpresidue_Driver(void *data){//finished
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
            flogf(stderr, "mpi_lgpresidue: dest=%d was distributed %d tasks\n", dest, total_count);
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
            MPI_Irecv(sg_result + i, 1, MPI_DOUBLE, dest, NUM_OF_TAGS * s_count + RESULT_TAG, MPI_COMM_WORLD, result_status + s_count);
        }
        MPI_Waitall(temp_sc, result_status, MPI_STATUSES_IGNORE);
    }
}  

void *wrapper_gpresidue_Receiver(void *data){//finished
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
            MPI_Isend(sg_result + ind, 1, MPI_DOUBLE, dest, NUM_OF_TAGS * locali + RESULT_TAG, MPI_COMM_WORLD, result_status + locali);
        }
        MPI_Waitall(s_count, result_status, MPI_STATUSES_IGNORE);
        sem_wait(&i_lock);
        MPI_Recv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
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


void *wrapper_gpresidue_Worker(void *data){//finished
    int core_id = (long) data;
    //stick_to_core(core_id);
    int ind, locali, ipair, i, j, k;
	double r1, r2;
    NP_MATRIX temp_wrapper;
    NP_MATRIX *temp_smikijsm, *temp_smijkjsm, *temp_smijiksm, *temp_smkjijsm, *temp_flip_result, *temp_ddot_result, *A, *B, *C, *D;
    temp_smikijsm = NULL;
    temp_smijkjsm = NULL;
    temp_smijiksm = NULL;
    temp_smkjijsm = NULL;
    temp_flip_result = NULL;
    temp_ddot_result = NULL;
    A = NULL;
    B = NULL;
    C = NULL;
    D = NULL;
	
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
            k = ipair / sg_mfno;
            j = ipair % sg_mfno;
            r1 = 0.0;
            elogf("original index=%d\n", ind);
            
            for (i = 0; i < sg_mfno; ++i){
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0) ||
                    (i < j && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + j) < 0) ||
                    (j < i && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + i) < 0)){
                    continue;
                }
				generate_supermatrix(&temp_smijiksm, i, j, i, k, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                generate_supermatrix(&temp_smikijsm, i, k, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                
                if (sg_nonredundant == 1 && i > k){
                    int idxki = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxki], sg_m_tdi[idxki], sg_n_tdi[idxki], sg_tra_tdi[idxki]);
                    
                    flip_ij(&temp_flip_result, k, i, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&temp_ddot_result, temp_smijiksm, temp_flip_result);
                    
                    
                } else{
                    int idxik = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxik], sg_m_tdi[idxik], sg_n_tdi[idxik], sg_tra_tdi[idxik]);
                    ddot_np_matrix(&temp_ddot_result, temp_smijiksm, &temp_wrapper);
                }
				ddot_np_matrix(&A, temp_ddot_result, temp_smikijsm);
				
				
				if (sg_nonredundant == 1 && i > j){
                    int idxji = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxji], sg_m_tbar[idxji], sg_n_tbar[idxji], sg_tra_tbar[idxji]); 
					flip_ij(&temp_flip_result, j, i, &temp_wrapper, sg_t_dim);
					temp_flip_result->m *= temp_flip_result->n;
					temp_flip_result->n = 1;
					A->n *= A->m;
					A->m = 1;
                    ddot_np_matrix(&C, A, temp_flip_result);
                    
                } else{
                    int idxij = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxij], sg_m_tbar[idxij] * sg_n_tbar[idxij], 1, sg_tra_tbar[idxij]); 
					A->n *= A->m;
					A->m = 1;
					ddot_np_matrix(&C, A, &temp_wrapper);
                }
				r1 += index_np_matrix(C, 0, 0);
			}
			
			
			
			i = ipair / sg_mfno;
            k = ipair % sg_mfno;
			r2 = 0.0;
            for (j = 0; j < sg_mfno; ++j){
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0) ||
                    (i < j && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + j) < 0) ||
                    (j < i && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + i) < 0)){
                    continue;
                }
				generate_supermatrix(&temp_smijkjsm, i, j, k, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                generate_supermatrix(&temp_smkjijsm, k, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                
                if (sg_nonredundant == 1 && k > j){
                    int idxjk = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxjk], sg_m_tdi[idxjk], sg_n_tdi[idxjk], sg_tra_tdi[idxjk]);
                    
                    flip_ij(&temp_flip_result, j, k, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&temp_ddot_result, temp_smijkjsm, temp_flip_result);
                    
                    
                } else{
                    int idxkj = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxkj], sg_m_tdi[idxkj], sg_n_tdi[idxkj], sg_tra_tdi[idxkj]);
                    ddot_np_matrix(&temp_ddot_result, temp_smijkjsm, &temp_wrapper);
                }
				ddot_np_matrix(&B, temp_ddot_result, temp_smkjijsm);
				
				
				if (sg_nonredundant == 1 && i > j){
                    int idxji = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxji], sg_m_tbar[idxji], sg_n_tbar[idxji], sg_tra_tbar[idxji]); 
					flip_ij(&temp_flip_result, j, i, &temp_wrapper, sg_t_dim);
					temp_flip_result->m *= temp_flip_result->n;
					temp_flip_result->n = 1;
					B->n *= B->m;
					B->m = 1;
                    ddot_np_matrix(&D, B, temp_flip_result);
                    
                } else{
                    int idxij = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_bar[idxij], sg_m_tbar[idxij] * sg_n_tbar[idxij], 1, sg_tra_tbar[idxij]); 
					B->n *= B->m;
					B->m = 1;
					ddot_np_matrix(&D, B, &temp_wrapper);
                }
				r2 += index_np_matrix(D, 0, 0);
			}
			//flogf(stderr, "Trying to write sg_result\n");
			sg_result[ind] = r1 + r2;
			//flogf(stderr, "Finished writing sg_result\n");

            sem_post(p_lock + locali); 
        } else if (sg_count < 0){
            sem_post(&i_lock);

            delete_np_matrix(&temp_smikijsm);
            delete_np_matrix(&temp_smijkjsm);
            delete_np_matrix(&temp_smijiksm);
            delete_np_matrix(&temp_smkjijsm);
            delete_np_matrix(&temp_ddot_result);
            delete_np_matrix(&temp_flip_result);
            delete_np_matrix(&A);
            delete_np_matrix(&B);
            delete_np_matrix(&C);
            delete_np_matrix(&D);
            break;
        } else {
            sem_post(&i_lock);
        }
    }
    pthread_exit(NULL); 
}

void mpi_lgpresidue(int len_pl_tdi, 
                    int* pairlist, 
                    double** T_di, int* m_tdi, int* n_tdi, char* tra_tdi, 
                    double** T_bar, int* m_tbar, int* n_tbar, char* tra_tbar,
                    int len_s_f,
                    double** S_matrix, int *m_sm, int *n_sm, char* tra_sm,
                    int len_tdim, int* t_dim,
                    int len_pll, int* pairlist_local, double* result,
                    int mfno, int nonredundant){
    double stime = omp_get_wtime();  
	int signal = SIGNAL_LGPRESIDUE;  
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }    
	
	//Bcast everything
	MPI_Bcast(&len_pl_tdi  , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("len_pl_tdi=%d\n", len_pl_tdi);
	MPI_Bcast(&len_s_f     , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("len_s_f=%d\n", len_s_f);
	MPI_Bcast(&len_tdim    , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("len_tdim=%d\n", len_tdim);
	MPI_Bcast(&len_pll    , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("len_tdim=%d\n", len_tdim);	

	if (WORLD_RANK != 0){
		pairlist = malloc(sizeof(int) * len_pl_tdi);
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
		t_dim = malloc(sizeof(int) * len_tdim);
		T_di = malloc(sizeof(double*) * len_pl_tdi);
		T_bar = malloc(sizeof(double*) * len_pl_tdi);
		result = malloc(sizeof(double) * len_pll);
		S_matrix = malloc(sizeof(double*) * len_s_f);
	}
	
	
	MPI_Bcast(pairlist     , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(pairlist_local,len_pll, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(m_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_tdi      , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_tbar       , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_tbar       , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_tbar     , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(result       , len_pl_tdi, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(m_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(n_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(tra_sm       , len_s_f   , MPI_CHAR, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(t_dim        , len_tdim  , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&mfno        , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	MPI_Bcast(&nonredundant, 1            , MPI_INT , 0, MPI_COMM_WORLD);

	
	flogf(stderr, "mpi_lgpresidue: %f seconds for all Bcasts to finish on processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
	
	sg_len_pl_tdi = len_pl_tdi;
	sg_pairlist = pairlist;       ///ASSUME PAIRLIST IS SORTED!!!
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
	
	sg_len_tdim = len_tdim;
	sg_t_dim =t_dim;
	
	sg_len_pll = len_pll;
	sg_result = result;
	sg_pairlist_local = pairlist_local;       ///ASSUME PAIRLIST IS SORTED!!!
	
	sg_mfno = mfno;
	sg_nonredundant = nonredundant;
	sg_i = 0;   
	
	
	MPI_Request *tdi_status, *sm_status, *tbar_status;
	tdi_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	tbar_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	sm_status = malloc(sizeof(MPI_Request) * len_s_f);

    int i, j, *dest;
    sem_init(&i_lock, 0, 1);
	
	//Root:
	if (WORLD_RANK == 0){ 

		flogf(stderr, "mpi_lgpresidue: total %d tasks\n", sg_len_pl_tdi);

		dest = malloc(sizeof(int) * (WORLD_SIZE - 1));
		pthread_t *driving_threads = malloc(sizeof(pthread_t) * NUM_OF_WTHREADS * (WORLD_SIZE - 1));
		
		pthread_t computing_threads[NUM_OF_CTHREADS];    
			 
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_create(computing_threads + i, NULL, wrapper_gpresidue_Local, (void*)i);
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
				pthread_create(driving_threads + i, NULL, wrapper_gpresidue_Driver, dest + i);
			}
		}
		
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_join(computing_threads[i], NULL);
		}
		flogf(stderr, "mpi_lgpresidue: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
		
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
        pthread_create(&receiving_thread, NULL, wrapper_gpresidue_Receiver, NULL);
        for (i = 0; i < NUM_OF_WTHREADS; ++i){
            pthread_create(working_threads + i, NULL, wrapper_gpresidue_Worker, (void*) i);
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
		free(result);
		free(pairlist);
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

		free(t_dim);
		
		free(T_di);
		free(S_matrix);

	}
	free(tdi_status);
	free(tbar_status);
	free(sm_status);
	
    sem_destroy(&i_lock);
	flogf(stderr, "mpi_lgpresidue: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
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
