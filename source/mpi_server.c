#include "mpi_helper.h"
#include "omp_helper.h"


MPI_Datatype MPI_MATRIX_META;
int WORLD_SIZE, NAME_LEN, WORLD_RANK, SERVER_SIZE, SERVER_RANK, CLIENT_SIZE, CLIENT_RANK;
char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];

void mpi_start_server(){
    MPI_Comm MPI_COMM_CLIENT, MPI_COMM_SERVER;
    MPI_Status status;
    char port_name[MPI_MAX_PORT_NAME];
    int again, continued;
    
	int provided_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided_level);
    MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_SERVER);
    MPI_Comm_size(MPI_COMM_SERVER, &SERVER_SIZE);
    MPI_Comm_rank(MPI_COMM_SERVER, &SERVER_RANK);
    MPI_Get_processor_name(PROCESSOR_NAME, &NAME_LEN);
    printf("[%d/%d]: Initialized\n", SERVER_RANK, SERVER_SIZE);
    
	if (SERVER_RANK == 0){
	    MPI_Open_port(MPI_INFO_NULL, port_name);
	    MPI_Publish_name("carbon_mpi", MPI_INFO_NULL, port_name);
	    printf("server available at %s\n",port_name);	
	    continued = TRUE;
	}
    
    continued = TRUE;
    while (continued){
        printf("Waiting for connnection\n");
        MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &MPI_COMM_CLIENT);
        MPI_Comm_size(MPI_COMM_CLIENT, &CLIENT_SIZE);
        MPI_Comm_rank(MPI_COMM_CLIENT, &CLIENT_RANK);
        printf("[%d/%d]: has [%d/%d] in MPI_COMM_CLIENT\n", SERVER_RANK, SERVER_SIZE, CLIENT_RANK, CLIENT_SIZE);
        again = TRUE;
        while (again){
            if (SERVER_RANK == 0){
                MPI_Recv(&again, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_CLIENT, &status);
            }
            
            
            MPI_Bcast(&again, 1, MPI_INT, 0, MPI_COMM_SERVER);
            switch (again){
                case 0:{
                    MPI_Comm_disconnect(&MPI_COMM_CLIENT);
                    again = FALSE;
                    continued = FALSE;
                    break;
                }
                case 1:{
                    MPI_Comm_disconnect(&MPI_COMM_CLIENT);                
                    again = FALSE;
                    break;
                }
                default:{
                    printf("[%d/%d]: received message %d\n", SERVER_RANK, SERVER_SIZE, again);
                    break;
                }
                    
            }
        }
    }
    
    if (SERVER_RANK == 0){
        MPI_Unpublish_name("carbon_mpi", MPI_INFO_NULL, port_name);
        MPI_Close_port(port_name);
    }
    MPI_Finalize();  
}
    
int main(){
    mpi_start_server();
    return 0;
}

