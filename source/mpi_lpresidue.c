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

static int sg_len_pl_tdi, sg_len_s_f, sg_len_tdim, sg_mfno, sg_nonredundant;
static char *sg_tra_tdi, *sg_tra_result, *sg_tra_sm, *sg_tra_fm;
static int *sg_completed, *sg_pairlist, *sg_m_tdi, *sg_n_tdi, *sg_m_result, *sg_n_result, *sg_m_sm, *sg_n_sm, *sg_m_fm, *sg_n_fm, *sg_t_dim;
static double **sg_T_di, **sg_result, **sg_S_matrix, **sg_F_matrix;

static NP_MATRIX sg_np_lf;
//static NP_MATRIX **sg_result_matrix;

extern MPI_Datatype MPI_MATRIX_META;
extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];

int intbsearch(int *arr, int length, int x){ 
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



void *wrapper_presidue_Local(void *data){
    int core_id = (long) data;
    stick_to_core(core_id);
    int ind, ipair, i, j, k;
    int s_i[CGRAPE_SIZE] = {0};
    int s_count = 0;
    int temp_smijijfm_ready = 0;
    NP_MATRIX temp_wrapper;
    NP_MATRIX *temp_smijijfm, *temp_smikijsm, *temp_smijkjsm, *temp_smijiksm, *temp_smkjijsm, *temp_flip_result, *temp_ddot_result, *ptr2wrapper, *A, *B, *C, *D;
    temp_smijijfm = NULL;
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
            
            while (sg_i < sg_len_pl_tdi && sg_completed[sg_i] == 1){
                ++sg_i;
            }
                
            if (sg_i >= sg_len_pl_tdi){
                sem_post(&i_lock);
                break;
            } else {
                s_i[s_count++] = sg_i++;
            }
        }
        sem_post(&i_lock);
        while (s_count > 0){
            ind = s_i[--s_count];
			
            ///Start actual computation
            ipair = sg_pairlist[ind];
            i = ipair / sg_mfno;
            j = ipair % sg_mfno;
            for (k = 0; k < sg_mfno; ++k){
                
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0)){
                    continue;
                }
                
                generate_supermatrix(&temp_smikijsm, i, k, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (k == j){
                    generate_supermatrix(&temp_smijijfm, i, j, i, j, sg_F_matrix, sg_t_dim, sg_mfno, sg_tra_fm);
                    temp_smijijfm_ready = 1;
                    axpy_np_matrix(&B, temp_smikijsm, -1.0 * index_np_matrix(&sg_np_lf, k, j), temp_smijijfm);
                    
                } else{
                    temp_smijijfm_ready = 0;
                    scal_np_matrix(&B, temp_smikijsm, -1.0 * index_np_matrix(&sg_np_lf, k, j));
                }
                
                generate_supermatrix(&temp_smijkjsm, i, j, k, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (i == k){
                    if (temp_smijijfm_ready != 1){
                        generate_supermatrix(&temp_smijijfm, i, j, i, j, sg_F_matrix, sg_t_dim, sg_mfno, sg_tra_fm);
                    }
                    axpy_np_matrix(&C, temp_smijkjsm, -1.0 * index_np_matrix(&sg_np_lf, i, k), temp_smijijfm);
                } else{
                    scal_np_matrix(&C, temp_smijkjsm, -1.0 * index_np_matrix(&sg_np_lf, i, k));
                }
                
                generate_supermatrix(&temp_smijiksm, i, j, i, k, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (sg_nonredundant == 1 && i > k){
                    int idxki = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxki], sg_m_tdi[idxki], sg_n_tdi[idxki], sg_tra_tdi[idxki]);
                    
                    flip_ij(&temp_flip_result, k, i, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&A, temp_smijiksm, temp_flip_result);
                    
                    
                } else{
                    int idxik = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxik], sg_m_tdi[idxik], sg_n_tdi[idxik], sg_tra_tdi[idxik]);
                    ddot_np_matrix(&A, temp_smijiksm, &temp_wrapper);
                }
                
                generate_supermatrix(&temp_smkjijsm, k, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (sg_nonredundant == 1 && k > j){
                    int idxjk = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxjk], sg_m_tdi[idxjk], sg_n_tdi[idxjk], sg_tra_tdi[idxjk]); 
                    flip_ij(&temp_flip_result, j, k, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&D, temp_flip_result, temp_smkjijsm);
                    
                } else{
                    int idxkj = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxkj], sg_m_tdi[idxkj], sg_n_tdi[idxkj], sg_tra_tdi[idxkj]); 
                    ddot_np_matrix(&D, &temp_wrapper, temp_smkjijsm);
                }
                
                elogf("final ddot calculation");
                wrap_np_matrix(&temp_wrapper, sg_result[ind], sg_m_result[ind], sg_n_result[ind], sg_tra_result[ind]);
                ptr2wrapper = &temp_wrapper;
                ddot_np_matrix(&temp_ddot_result, A, B);
                axpy_np_matrix(&ptr2wrapper, temp_ddot_result, 1.0, ptr2wrapper);
                ddot_np_matrix(&temp_ddot_result, C, D);
                axpy_np_matrix(&ptr2wrapper, temp_ddot_result, 1.0, ptr2wrapper);
                
            }
            ///End actual computation
        }       
        
        if (sg_i >= sg_len_pl_tdi){
            delete_np_matrix(&temp_smijijfm);
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

void *wrapper_presidue_Driver(void *data){//finished
    int total_count = 0;
    int i, temp_sc, s_count;
    int dest = *(int *)data;
    MPI_Request result_status[WGRAPE_SIZE], signal_status;
    int s_i[WGRAPE_SIZE] = {0};
    while (TRUE){
        s_count = 0;
        sem_wait(&i_lock);
        
        while (s_count < WGRAPE_SIZE){
            
            while (sg_i < sg_len_pl_tdi && sg_completed[sg_i] == 1){
                ++sg_i;
            }
                
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
            fprintf(stderr, "mpi_lpresidue: dest=%d was distributed %d tasks\n", dest, total_count);
            MPI_Isend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
            MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
            pthread_exit(NULL); 
        } 
        sem_post(&i_lock);
        MPI_Isend(&s_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
        
        MPI_Isend(s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
        
        temp_sc = s_count;
        
        while (s_count > 0){
            --s_count;
            i = s_i[s_count];
            MPI_Irecv(sg_result[i], sg_m_result[i] * sg_n_result[i], MPI_DOUBLE, dest, NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD, result_status + s_count);
        }
        MPI_Waitall(temp_sc, result_status, MPI_STATUSES_IGNORE);
    }
}  

void *wrapper_presidue_Receiver(void *data){//finished
    int locali, ind;
    int s_count = 0;
    int dest = 0; // to root
	//int s_i[WGRAPE_SIZE];
    MPI_Request meta_status[WGRAPE_SIZE], result_status[WGRAPE_SIZE], signal_status;
    
    sem_wait(&i_lock);
    MPI_Irecv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
    MPI_Wait(&signal_status, MPI_STATUS_IGNORE);
    s_count = sg_count;
    sem_post(&i_lock);
	
    while (s_count > 0){
		
		MPI_Irecv(sg_s_i, s_count, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
		MPI_Wait(&signal_status, MPI_STATUS_IGNORE);	
		
        for (locali = s_count - 1; locali >= 0; --locali){
            //ind = sg_s_i[locali];
            //sg_result[locali] = realloc(sg_result[locali], sizeof(double) * sg_m_result[ind] * sg_n_result[ind]);  
            sem_post(&t_lock);  
            sem_post(r_lock + locali); 
        }

        for (locali = s_count - 1; locali >= 0; --locali){
			ind = sg_s_i[locali];
            sem_wait(p_lock + locali);
            MPI_Isend(sg_result[ind], sg_m_result[ind] * sg_n_result[ind], MPI_DOUBLE, dest, NUM_OF_TAGS * ind + RESULT_TAG, MPI_COMM_WORLD, result_status + locali);
        }
        MPI_Waitall(s_count, result_status, MPI_STATUSES_IGNORE);
        sem_wait(&i_lock);
        MPI_Irecv(&sg_count, 1, MPI_INT, dest, SIGNAL_TAG, MPI_COMM_WORLD, &signal_status);
        MPI_Wait(&signal_status, MPI_STATUS_IGNORE);   
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


void *wrapper_presidue_Worker(void *data){//finished
    int core_id = (long) data;
    stick_to_core(core_id);
    int ind, locali, ipair, i, j, k;
    int temp_smijijfm_ready = 0;
    NP_MATRIX temp_wrapper;
    NP_MATRIX *temp_smijijfm, *temp_smikijsm, *temp_smijkjsm, *temp_smijiksm, *temp_smkjijsm, *temp_flip_result, *temp_ddot_result, *ptr2wrapper, *A, *B, *C, *D;
    temp_smijijfm = NULL;
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
            ipair = sg_pairlist[ind];
            i = ipair / sg_mfno;
            j = ipair % sg_mfno;
            
            elogf("original index=%d\n", ind);
            
            for (k = 0; k < sg_mfno; ++k){
                if ((k < i && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + i) < 0) ||
                    (i < k && intbsearch(sg_pairlist, sg_len_pl_tdi, i * sg_mfno + k) < 0) ||
                    (k < j && intbsearch(sg_pairlist, sg_len_pl_tdi, k * sg_mfno + j) < 0) ||
                    (j < k && intbsearch(sg_pairlist, sg_len_pl_tdi, j * sg_mfno + k) < 0)){
                    continue;
                }
                generate_supermatrix(&temp_smikijsm, i, k, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (k == j){
                    generate_supermatrix(&temp_smijijfm, i, j, i, j, sg_F_matrix, sg_t_dim, sg_mfno, sg_tra_fm);
                    temp_smijijfm_ready = 1;
                    axpy_np_matrix(&B, temp_smikijsm, -1.0 * index_np_matrix(&sg_np_lf, k, j), temp_smijijfm);
                    
                } else{
                    temp_smijijfm_ready = 0;
                    scal_np_matrix(&B, temp_smikijsm, -1.0 * index_np_matrix(&sg_np_lf, k, j));
                }
                
                generate_supermatrix(&temp_smijkjsm, i, j, k, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (i == k){
                    if (temp_smijijfm_ready != 1){
                        generate_supermatrix(&temp_smijijfm, i, j, i, j, sg_F_matrix, sg_t_dim, sg_mfno, sg_tra_fm);
                    }
                    axpy_np_matrix(&C, temp_smijkjsm, -1.0 * index_np_matrix(&sg_np_lf, i, k), temp_smijijfm);
                } else{
                    scal_np_matrix(&C, temp_smijkjsm, -1.0 * index_np_matrix(&sg_np_lf, i, k));
                }
                
                generate_supermatrix(&temp_smijiksm, i, j, i, k, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (sg_nonredundant == 1 && i > k){
                    int idxki = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + i);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxki], sg_m_tdi[idxki], sg_n_tdi[idxki], sg_tra_tdi[idxki]);
                    
                    flip_ij(&temp_flip_result, k, i, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&A, temp_smijiksm, temp_flip_result);
                    
                    
                } else{
                    int idxik = intbsearch(sg_pairlist, sg_len_pl_tdi, i*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxik], sg_m_tdi[idxik], sg_n_tdi[idxik], sg_tra_tdi[idxik]);
                    ddot_np_matrix(&A, temp_smijiksm, &temp_wrapper);
                }
                
                generate_supermatrix(&temp_smkjijsm, k, j, i, j, sg_S_matrix, sg_t_dim, sg_mfno, sg_tra_sm);
                if (sg_nonredundant == 1 && k > j){
                    int idxjk = intbsearch(sg_pairlist, sg_len_pl_tdi, j*sg_mfno + k);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxjk], sg_m_tdi[idxjk], sg_n_tdi[idxjk], sg_tra_tdi[idxjk]); 
                    flip_ij(&temp_flip_result, j, k, &temp_wrapper, sg_t_dim);
                    ddot_np_matrix(&D, temp_flip_result, temp_smkjijsm);
                    
                } else{
                    int idxkj = intbsearch(sg_pairlist, sg_len_pl_tdi, k*sg_mfno + j);
                    wrap_np_matrix(&temp_wrapper, sg_T_di[idxkj], sg_m_tdi[idxkj], sg_n_tdi[idxkj], sg_tra_tdi[idxkj]); 
                    ddot_np_matrix(&D, &temp_wrapper, temp_smkjijsm);
                }
                
             
                wrap_np_matrix(&temp_wrapper, sg_result[ind], sg_m_result[ind], sg_n_result[ind], sg_tra_result[ind]);
                ptr2wrapper = &temp_wrapper;
                ddot_np_matrix(&temp_ddot_result, A, B);
                axpy_np_matrix(&ptr2wrapper, temp_ddot_result, 1.0, ptr2wrapper);
                ddot_np_matrix(&temp_ddot_result, C, D);
                axpy_np_matrix(&ptr2wrapper, temp_ddot_result, 1.0, ptr2wrapper);
				
            }

            sem_post(p_lock + locali); 
        } else if (sg_count < 0){
            sem_post(&i_lock);

            delete_np_matrix(&temp_smijijfm);
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

void mpi_lpresidue(int len_pl_tdi, 
                   int* completed, int* pairlist, 
                   double** T_di, int* m_tdi, int* n_tdi, char* tra_tdi, 
                   double** result, int* m_result, int* n_result, char* tra_result,
                   int len_s_f,
                   double** S_matrix, int *m_sm, int *n_sm, char* tra_sm,
                   double** F_matrix, int *m_fm, int *n_fm, char* tra_fm,
                   double* loc_fork, int m_lf, int n_lf, char tra_lf,
                   int len_tdim, int* t_dim,
                   int mfno, int nonredundant){
    double stime = omp_get_wtime();  
	int signal = SIGNAL_LPRESIDUE;  
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
	MPI_Bcast(&m_lf         , 1   , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("m_lf=%d\n", m_lf);
	MPI_Bcast(&n_lf         , 1   , MPI_INT , 0, MPI_COMM_WORLD);	
	elogf("n_lf=%d\n", n_lf);
	if (WORLD_RANK != 0){
		completed = malloc(sizeof(int) * len_pl_tdi);
		pairlist = malloc(sizeof(int) * len_pl_tdi);
		m_tdi = malloc(sizeof(int) * len_pl_tdi);
		n_tdi = malloc(sizeof(int) * len_pl_tdi);
		tra_tdi = malloc(sizeof(char) * len_pl_tdi);
		m_result = malloc(sizeof(int) * len_pl_tdi);
		n_result = malloc(sizeof(int) * len_pl_tdi);
		tra_result = malloc(sizeof(char) * len_pl_tdi);
		m_sm = malloc(sizeof(int) * len_s_f);
		n_sm = malloc(sizeof(int) * len_s_f);
		tra_sm = malloc(sizeof(char) * len_s_f);
		m_fm = malloc(sizeof(int) * len_s_f);
		n_fm = malloc(sizeof(int) * len_s_f);
		tra_fm = malloc(sizeof(char) * len_s_f);
		t_dim = malloc(sizeof(int) * len_tdim);
		loc_fork = malloc(sizeof(double) * m_lf * n_lf);
		T_di = malloc(sizeof(double*) * len_pl_tdi);
		result = malloc(sizeof(double*) * len_pl_tdi);
		S_matrix = malloc(sizeof(double*) * len_s_f);
		F_matrix = malloc(sizeof(double*) * len_s_f);
	}
	
	MPI_Bcast(completed    , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("completed[%d]=%d\n", len_pl_tdi - 1, completed[len_pl_tdi - 1]);
	MPI_Bcast(pairlist     , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("pairlist[%d]=%d\n", len_pl_tdi - 1, pairlist[len_pl_tdi - 1]);
	MPI_Bcast(m_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("m_tdi[%d]=%d\n", len_pl_tdi - 1, m_tdi[len_pl_tdi - 1]);
	MPI_Bcast(n_tdi        , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("n_tdi[%d]=%d\n", len_pl_tdi - 1, n_tdi[len_pl_tdi - 1]);
	MPI_Bcast(tra_tdi      , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	elogf("tra_tdi[%d]=%c\n", len_pl_tdi - 1, tra_tdi[len_pl_tdi - 1]);
	MPI_Bcast(m_result     , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("m_result[%d]=%d\n", len_pl_tdi - 1, m_result[len_pl_tdi - 1]);
	MPI_Bcast(n_result     , len_pl_tdi, MPI_INT , 0, MPI_COMM_WORLD);
	elogf("n_result[%d]=%d\n", len_pl_tdi - 1, n_result[len_pl_tdi - 1]);
	MPI_Bcast(tra_result   , len_pl_tdi, MPI_CHAR, 0, MPI_COMM_WORLD);
	elogf("tra_result[%d]=%c\n", len_pl_tdi - 1, tra_result[len_pl_tdi - 1]);
	
	MPI_Bcast(m_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("m_sm[%d]=%d\n", len_s_f - 1, m_sm[len_s_f - 1]);
	MPI_Bcast(n_sm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("n_sm[%d]=%d\n", len_s_f - 1, n_sm[len_s_f - 1]);
	MPI_Bcast(tra_sm       , len_s_f   , MPI_CHAR, 0, MPI_COMM_WORLD);
	elogf("tra_sm[%d]=%c\n", len_s_f - 1, tra_sm[len_s_f - 1]);
	MPI_Bcast(m_fm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("m_fm[%d]=%d\n", len_s_f - 1, m_fm[len_s_f - 1]);
	MPI_Bcast(n_fm         , len_s_f   , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("n_fm[%d]=%d\n", len_s_f - 1, n_fm[len_s_f - 1]);
	MPI_Bcast(tra_fm       , len_s_f   , MPI_CHAR, 0, MPI_COMM_WORLD);
	elogf("tra_fm[%d]=%c\n", len_s_f - 1, tra_fm[len_s_f - 1]);
	
	MPI_Bcast(t_dim        , len_tdim  , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("t_dim[%d]=%d\n", len_tdim - 1, t_dim[len_tdim - 1]);
	MPI_Bcast(&mfno        , 1            , MPI_INT , 0, MPI_COMM_WORLD);
	elogf("mfno=%d\n", mfno);
	MPI_Bcast(&nonredundant, 1            , MPI_INT , 0, MPI_COMM_WORLD);
    elogf("nonredundant=%d\n", nonredundant);

	MPI_Bcast(&tra_lf       , 1   , MPI_CHAR, 0, MPI_COMM_WORLD);	
	elogf("tra_lf=%c\n", tra_lf);
	MPI_Bcast(loc_fork, m_lf * n_lf, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	fprintf(stderr, "mpi_lpresidue: %f seconds for all Bcasts to finish on processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
	
	sg_len_pl_tdi = len_pl_tdi;
	sg_completed = completed;
	sg_pairlist = pairlist;       ///ASSUME PAIRLIST IS SORTED!!!
	sg_T_di = T_di;//sendlatter
	sg_m_tdi = m_tdi;
	sg_n_tdi = n_tdi;
	sg_tra_tdi = tra_tdi;
	sg_result = result;//NEED TO BE SENT TO WORKER
	sg_m_result = m_result;
	sg_n_result = n_result;
	sg_tra_result = tra_result;
	
	sg_len_s_f = len_s_f;
	sg_S_matrix = S_matrix;//sendlatter
	sg_m_sm = m_sm;
	sg_n_sm = n_sm;
	sg_tra_sm = tra_sm;
	sg_F_matrix = F_matrix;//sendlatter
	sg_m_fm = m_fm;
	sg_n_fm = n_fm;
	sg_tra_fm = tra_fm;
	
	sg_len_tdim = len_tdim;
	sg_t_dim =t_dim;
	
	sg_mfno = mfno;
	sg_nonredundant = nonredundant;
	sg_i = 0;   
	
	
	wrap_np_matrix(&sg_np_lf, loc_fork, m_lf, n_lf, tra_lf);//send separately for now
	MPI_Request *result_status, *tdi_status, *sm_status, *fm_status;
	result_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	tdi_status = malloc(sizeof(MPI_Request) * len_pl_tdi);
	sm_status = malloc(sizeof(MPI_Request) * len_s_f);
	fm_status = malloc(sizeof(MPI_Request) * len_s_f);

    int i, j, *dest;
    sem_init(&i_lock, 0, 1);
	
	//Root:
	if (WORLD_RANK == 0){ 

		fprintf(stderr, "mpi_lpresidue: total %d tasks\n", sg_len_pl_tdi);

		dest = malloc(sizeof(int) * (WORLD_SIZE - 1));
		pthread_t *driving_threads = malloc(sizeof(pthread_t) * NUM_OF_WTHREADS * (WORLD_SIZE - 1));
		
		pthread_t computing_threads[NUM_OF_CTHREADS];    
			 
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_create(computing_threads + i, NULL, wrapper_presidue_Local, (void*)i);
		}
		
		for (i = 0; i < WORLD_SIZE - 1; ++i){
			dest[i] = i + 1;
		}		
		
		
		
		for (i = 0; i < len_pl_tdi; ++i){
			for (j = 0; j < WORLD_SIZE - 1; j++){
				MPI_Isend(T_di[i], m_tdi[i] * n_tdi[i], MPI_DOUBLE, dest[j], NUM_OF_TAGS * i + META_TAG, MPI_COMM_WORLD, tdi_status + i);
				MPI_Isend(result[i], m_result[i] * n_result[i], MPI_DOUBLE, dest[j], NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD, result_status + i);
			}
		}
		
		if (WORLD_SIZE > 1){
			MPI_Waitall(len_pl_tdi, tdi_status, MPI_STATUSES_IGNORE);
			MPI_Waitall(len_pl_tdi, result_status, MPI_STATUSES_IGNORE);
		}
		fprintf(stderr, "%s: %s: %s\n", __FILE__, __func__, "sent T_di, raw result");
		
		for (i = 0; i < len_s_f; ++i){
			for (j = 0; j < WORLD_SIZE - 1; j++){
				MPI_Isend(S_matrix[i], m_sm[i] * n_sm[i], MPI_DOUBLE, dest[j], NUM_OF_TAGS * i + DATA_A_TAG, MPI_COMM_WORLD, sm_status + i);
				MPI_Isend(F_matrix[i], m_fm[i] * n_fm[i], MPI_DOUBLE, dest[j], NUM_OF_TAGS * i + DATA_B_TAG, MPI_COMM_WORLD, fm_status + i);
			}
		}
		
		if (WORLD_SIZE > 1){
			MPI_Waitall(len_s_f, sm_status, MPI_STATUSES_IGNORE);
			MPI_Waitall(len_s_f, fm_status, MPI_STATUSES_IGNORE);
		}
		
		
		
		if (NUM_OF_WTHREADS > 0){
			for (i = 0; i < WORLD_SIZE - 1; ++i){
				pthread_create(driving_threads + i, NULL, wrapper_presidue_Driver, dest + i);
			}
		}
		
		for (i = 0; i < NUM_OF_CTHREADS; ++i){
			pthread_join(computing_threads[i], NULL);
		}
		fprintf(stderr, "mpi_lpresidue: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
		
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
			MPI_Irecv(sg_T_di[i], temp_size, MPI_DOUBLE, 0, NUM_OF_TAGS * i + META_TAG, MPI_COMM_WORLD, tdi_status + i);
			
			temp_size = sg_m_result[i] * sg_n_result[i];
			sg_result[i] = malloc(sizeof(double) * temp_size);
			MPI_Irecv(sg_result[i], temp_size, MPI_DOUBLE, 0, NUM_OF_TAGS * i + RESULT_TAG, MPI_COMM_WORLD, result_status + i);			
		}
		
		for (i = 0; i < len_s_f; ++i){
			temp_size = sg_m_sm[i] * sg_n_sm[i];
			sg_S_matrix[i] = malloc(sizeof(double) * temp_size);
			MPI_Irecv(sg_S_matrix[i], temp_size, MPI_DOUBLE, 0, NUM_OF_TAGS * i + DATA_A_TAG, MPI_COMM_WORLD, sm_status + i);
			
			temp_size = sg_m_fm[i] * sg_n_fm[i];
			sg_F_matrix[i] = malloc(sizeof(double) * temp_size);
			MPI_Irecv(sg_F_matrix[i], temp_size, MPI_DOUBLE, 0, NUM_OF_TAGS * i + DATA_B_TAG, MPI_COMM_WORLD, fm_status + i);
		}
		
		MPI_Waitall(len_pl_tdi, tdi_status, MPI_STATUSES_IGNORE);
		MPI_Waitall(len_pl_tdi, result_status, MPI_STATUSES_IGNORE);
		MPI_Waitall(len_s_f, sm_status, MPI_STATUSES_IGNORE);
		MPI_Waitall(len_s_f, fm_status, MPI_STATUSES_IGNORE);
		
		//sg_result_matrix = malloc(sizeof(NP_MATRIX*) * WGRAPE_SIZE);
        sem_init(&t_lock, 0, 0); //task lock : how many unfinished task in current comm
		
        for (i = 0; i < WGRAPE_SIZE; ++i){
            sem_init(r_lock + i, 0, 0); //resource lock : if current task resource is ready
            sem_init(p_lock + i, 0, 0); //product lock : if current product is ready
        }
        
        pthread_t working_threads[NUM_OF_WTHREADS];
        pthread_t receiving_thread;
        pthread_create(&receiving_thread, NULL, wrapper_presidue_Receiver, NULL);
        for (i = 0; i < NUM_OF_WTHREADS; ++i){
            pthread_create(working_threads + i, NULL, wrapper_presidue_Worker, (void*) i);
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
		for (i = 0; i < WGRAPE_SIZE; ++i){
			//delete_np_matrix(sg_result_matrix + i);
			free(sg_result[i]);
		}
		//free(sg_result_matrix);
		
		for (i = 0; i < len_pl_tdi; ++i){
			free(sg_T_di[i]);
		}
		
		for (i = 0; i < len_s_f; ++i){
			free(sg_S_matrix[i]);
			free(sg_F_matrix[i]);
		}
		free(sg_result);
		free(completed);
		free(pairlist);
		free(m_tdi);
		free(n_tdi);
		free(tra_tdi);
		free(m_result);
		free(n_result);
		free(tra_result);
		free(m_sm);
		free(n_sm);
		free(tra_sm);
		free(m_fm);
		free(n_fm);
		free(tra_fm);
		free(t_dim);
		free(loc_fork);
		
		free(T_di);
		free(S_matrix);
		free(F_matrix);

	}
 	free(result_status);
	free(tdi_status);
	free(sm_status);
	free(fm_status);
	
    sem_destroy(&i_lock);
	fprintf(stderr, "mpi_lpresidue: %f seconds taken from processor %s, rank %d out of %d worlds.\n", 
        omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
}


void pth_lpresidue(int len_pl_tdi, 
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
        pthread_create(computing_threads + i, NULL, wrapper_presidue_Local, NULL);
    }
    
    for (i = 0; i < NUM_OF_CTHREADS; ++i){
        pthread_join(computing_threads[i], NULL);
    }
 	    
    sem_destroy(&i_lock);
    fprintf(stderr, "pth_lpresidue: %f seconds taken.\n", omp_get_wtime() - stime);
}
