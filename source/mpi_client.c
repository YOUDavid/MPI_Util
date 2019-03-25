#include "mpi_helper.h"
#include "omp_helper.h"


MPI_Datatype MPI_MATRIX_META;
int WORLD_SIZE, NAME_LEN, WORLD_RANK;
char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];

    
int main(){
    MPI_Comm server;
    int again;
    char port_name[MPI_MAX_PORT_NAME];
    MPI_Init(NULL, NULL);
    MPI_Lookup_name("carbon_mpi", MPI_INFO_NULL, port_name);
    MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &server);
    printf("Connected\n");
    do {
        scanf("%d", &again);
        printf("Got input %d\n", again);
        MPI_Send(&again, 1, MPI_INT, 0, 0, server);
    } while (again != TRUE && again != FALSE);
    MPI_Comm_disconnect( &server );
    MPI_Finalize();
    return 0;
}

