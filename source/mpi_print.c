#include "mpi_helper.h"   
#include "omp_helper.h" 

extern int WORLD_SIZE, NAME_LEN, WORLD_RANK;
extern char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];

void mpi_print(int len, double* arr) {
    int signal = SIGNAL_PRINT; //0 for mpi_print

    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    int i, j;
    int slice = 8;
    int totsiz = 32768 * 32768;
    MPI_Request *signal_status = malloc(sizeof(MPI_Request) * 8192);
    double *marr = malloc(sizeof(double) * 32768 * 32768);
    double stime;
    
    if (WORLD_RANK == 0){
        totsiz = 32768 * 32768 / slice;
        stime = omp_get_wtime(); 
        for (j = 0; j < slice; ++j){
            MPI_Isend(marr + j * totsiz, totsiz, MPI_DOUBLE, 1, j, MPI_COMM_WORLD, signal_status + j);
        }
        fprintf(stderr, "mpi_print: %f seconds taken to MPI_Isend with %d slice(s), from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, slice, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
        stime = omp_get_wtime(); 
        MPI_Waitall(slice, signal_status, MPI_STATUSES_IGNORE);
        fprintf(stderr, "mpi_print: %f seconds taken to finish sending with %d slice(s), from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, slice, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);

    } else {

        totsiz = 32768 * 32768 / slice;
        stime = omp_get_wtime(); 
        for (j = 0; j < slice; ++j){
            MPI_Irecv(marr + j * totsiz, totsiz, MPI_DOUBLE, 0, j, MPI_COMM_WORLD, signal_status + j);
        }
        fprintf(stderr, "mpi_print: %f seconds taken to MPI_Irecv with %d slice(s), from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, slice, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
        stime = omp_get_wtime(); 
        MPI_Waitall(slice, signal_status, MPI_STATUSES_IGNORE);
        fprintf(stderr, "mpi_print: %f seconds taken to finish receiving with %d slice(s), from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, slice, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);

    }
    free(marr);
    free(signal_status);
}






