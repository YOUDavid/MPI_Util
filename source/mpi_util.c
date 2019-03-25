#include "mpi_helper.h"   
#include "omp_helper.h"


MPI_Datatype MPI_MATRIX_META;
//MPI_Comm *WTHREAD_COMM;
int WORLD_SIZE, NAME_LEN, WORLD_RANK;
char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];
//double *TEMP_ARR;
//double **A_ARR, **B_ARR, **C_ARR;
void mpi_setONT(int ont) {
    int signal = SIGNAL_SETONT; //0 for mpi_print
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&ont, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    char onts[3];
    sprintf(onts, "%d", ont);
    setenv("OMP_NUM_THREADS", onts, 1);
    omp_set_num_threads(ont);
    //omp_set_max_threads(ont);
    //printf("OMP_NUM_THREADS: %s from cpu %3d in processor %s, rank %d out of %d processors/worlds \n", getenv("OMP_NUM_THREADS"), sched_getcpu(), PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
}
    


void mpi_init(){
    int i, signal; //chose which function to execute
	int provided_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided_level);
	
    MPI_Comm_size(MPI_COMM_WORLD , &WORLD_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Get_processor_name(PROCESSOR_NAME, &NAME_LEN);
    fprintf(stderr, "P%d: Initialized\n", WORLD_RANK);
	if (provided_level != MPI_THREAD_MULTIPLE){
		fprintf(stderr, "P%d: no multithreading support!\n", WORLD_RANK);
	}
/*	if (WORLD_RANK != 0){*/
/*        TEMP_ARR = malloc(sizeof(double) * 3 * CHUNCK_SIZE * WGRAPE_SIZE);*/
/*        A_ARR = malloc(sizeof(double*) * WGRAPE_SIZE); */
/*        B_ARR = malloc(sizeof(double*) * WGRAPE_SIZE); */
/*        C_ARR = malloc(sizeof(double*) * WGRAPE_SIZE); */
/*        for (i = 0; i < WGRAPE_SIZE; ++i){*/
/*            //A_ARR[i] = TEMP_ARR + CHUNCK_SIZE * (3 * i);*/
/*            //B_ARR[i] = TEMP_ARR + CHUNCK_SIZE * (3 * i + 1);*/
/*            //C_ARR[i] = TEMP_ARR + CHUNCK_SIZE * (3 * i + 2);*/
/*        }*/
/*    }*/
    /* create a type for struct MPI_MATRIX */
    int blocklengths[NBLOCKS_MATRIX_META] = {[0 ... NBLOCKS_MATRIX_META - 1] = 1};
    MPI_Datatype types[NBLOCKS_MATRIX_META] = {MPI_CHAR, MPI_CHAR, 
                                               MPI_INT, MPI_INT, MPI_INT, 
                                               MPI_INT, MPI_INT, MPI_INT, 
                                               MPI_INT, MPI_INT, MPI_INT,
                                               MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[NBLOCKS_MATRIX_META] = {offsetof(MATRIX_META, tra_a),
                                             offsetof(MATRIX_META, tra_b),
                                             offsetof(MATRIX_META, m),
                                             offsetof(MATRIX_META, n),
                                             offsetof(MATRIX_META, k),
                                             offsetof(MATRIX_META, lda),
                                             offsetof(MATRIX_META, ldb),
                                             offsetof(MATRIX_META, ldc),
                                             offsetof(MATRIX_META, offseta),
                                             offsetof(MATRIX_META, offsetb),
                                             offsetof(MATRIX_META, offsetc),
                                             offsetof(MATRIX_META, alpha),
                                             offsetof(MATRIX_META, beta)};    

    MPI_Type_create_struct(NBLOCKS_MATRIX_META, blocklengths, offsets, types, &MPI_MATRIX_META);
    MPI_Type_commit(&MPI_MATRIX_META);    
    /*WTHREAD_COMM = malloc(sizeof(MPI_Comm) * (WORLD_SIZE - 1));
    for (i = 0; i < WORLD_SIZE - 1; ++i){
        MPI_Comm_dup(MPI_COMM_WORLD, WTHREAD_COMM + i);
    }*/
    
    
    
    if (WORLD_RANK != 0){
        //In worker processes, block until root signals workers to continue
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        while (signal != SIGNAL_EXIT){
            switch (signal){
                case SIGNAL_LPRESIDUE:
                    mpi_lpresidue(0, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL, 
								  NULL, NULL, NULL, NULL,
                                  NULL, 0, 0, '\0', 0, NULL, 0, 0);    
                    break;
                case SIGNAL_LNPDGEMM:
                    mpi_lNPdgemm(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                 0, NULL);
                    break;
                case SIGNAL_LGPRESIDUE:
                    mpi_lgpresidue(0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                  NULL, 0, NULL, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL,
                                  0, 0);
                    break;
                case SIGNAL_LGENPRESIDUE:
                    mpi_lgenpresidue(0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                  0, NULL, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL,
                                  NULL, NULL, NULL, NULL, 0);
                    break;					
                case SIGNAL_SETONT:
                    mpi_setONT(0);
                    break;
                case SIGNAL_PRINT:
                    mpi_print(0, NULL);
                    break;
            }
            //After finishing a task, block again
            MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        fprintf(stderr, "P%d: Finalized\n", WORLD_RANK);
        /*for (i = 0; i < WORLD_SIZE - 1; ++i){
            MPI_Comm_free(WTHREAD_COMM + i);
        }*/
        
        MPI_Finalize();
        //free(WTHREAD_COMM);
    } else{
        //In root process, do nothing, just return

    }
}


void mpi_final(){
    int signal = SIGNAL_EXIT;
    int i;
    MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /*for (i = 0; i < NUM_OF_WTHREADS - 1; ++i){
        MPI_Comm_free(WTHREAD_COMM + i);
    }*/
    fprintf(stderr, "P%d: Finalized\n", WORLD_RANK);
    MPI_Finalize();
/*    free(TEMP_ARR);*/
/*    free(A_ARR);*/
/*    free(B_ARR);*/
/*    free(C_ARR);*/
    //free(WTHREAD_COMM);
    fprintf(stdout, "NUM_OF_CTHREADS:\t%d\n", NUM_OF_CTHREADS);
    fprintf(stdout, "NUM_OF_WTHREADS:\t%d\n", NUM_OF_WTHREADS);
    fprintf(stdout, "CGRAPE_SIZE:\t%d\n", CGRAPE_SIZE);
    fprintf(stdout, "WGRAPE_SIZE:\t%d\n", WGRAPE_SIZE);
}

