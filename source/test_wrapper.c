#define DEBUG 0
#include "shared_types.h"
#include "shared_library.h" 
#include "blas_helper.h"
#include "omp_helper.h" 
#include <mpi.h>

int WORLD_SIZE, NAME_LEN, WORLD_RANK;
char PROCESSOR_NAME[MPI_MAX_PROCESSOR_NAME];

static int sg_signal;
static int sg_local_indices[WGRAPE_SIZE];

void mpi_setONT(int ont) {
    int signal = SIGNAL_SETONT; 
    if (WORLD_RANK == 0){
        MPI_Bcast(&signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&ont, 1, MPI_INT, 0, MPI_COMM_WORLD);
    omp_set_num_threads(ont);
    char onts[6];
    sprintf(onts, "%d", ont);
    setenv("OMP_NUM_THREADS", onts, 1);
    fprintf(stderr, "OMP_NUM_THREADS: %s from cpu %3d in processor %s, rank %d out of %d processors/worlds \n", getenv("OMP_NUM_THREADS"), sched_getcpu(), PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
}
    



#define TYPE_CHAR 0
#define TYPE_INT 1
#define TYPE_DOUBLE 2
#define TYPE_POINTER 3
#define TYPE_NP_MATRIX 4



typedef struct Type{
    int currentType;
    int currentLength;
    void *currentPointer;
} Type;
void test_print_arr(Type *arr);

typedef struct Datapack{
    int allEntryNumber;
    Type *typeList;
    
    int sharedEntryNumber;
    int *sharedEntryIndices;
    
	int dependentEntryNumber;
    int *dependentEntryIndices;
    
    int resultEntryNumber;
    int *resultEntryIndices;
	
	int totalTaskNumber;
	char *isCompleted;
	unsigned long actualFunction;
} Datapack;
    
typedef struct Controlpack{
	int current_work_index;
    int current_work_count;
	int remote_indices[WGRAPE_SIZE];
    sem_t counter_lock, thread_lock, resource_lock[WGRAPE_SIZE], product_lock[WGRAPE_SIZE];
} Controlpack;
	
	
typedef struct Workpack{
    Controlpack *controlpack;
	Datapack *datapack;
} Workpack;


size_t Type_sizeof(Type* type){
	switch(type->currentType){
		case TYPE_CHAR:
			return sizeof(char);
		case TYPE_INT:
            return sizeof(int);
		case TYPE_DOUBLE:
            return sizeof(double);
		case TYPE_POINTER:{
            return sizeof(void*);
		}
	}    
	return sizeof(void*);
}

#define Type_print(type) elogf("Ln:%d, type: %p, currentPointer: %p, currentType: %d, currentLength: %d\n", __LINE__, (type), (type)->currentPointer, (type)->currentType, (type)->currentLength)


void *Type_get_pointer(Type* type, const long offset){
    Type_print(type);
	switch(type->currentType){
		case TYPE_CHAR:
    		elogf("Ln:%d\n", __LINE__);
			return ((char*)(type->currentPointer)) + offset;
		case TYPE_INT:
			return ((int*)(type->currentPointer)) + offset;
		case TYPE_DOUBLE:
			return ((double*)(type->currentPointer)) + offset;
		case TYPE_POINTER:
		    elogf("Ln:%d\n", __LINE__);
			return ((Type*)(type->currentPointer)) + offset;
	}
	printf("%s :unrecognized currentType!\n", __func__);
	return NULL;
}



void Type_receive_data_local(Type* type, const int destination){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_INT:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_DOUBLE:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_POINTER:{
		    int i;
		    for (i = type->currentLength - 1; i >= 0; --i){
    			Type_receive_data_local(((Type*) (type->currentPointer)) + i, destination);
		    }
		    break;
		}
	}
}

void Type_receive_data_local_offset(Type* type, const int destination, const long offset){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Recv((char*)Type_get_pointer(type, offset), 1, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_INT:
			MPI_Recv((int*)Type_get_pointer(type, offset), 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_DOUBLE:
			MPI_Recv((double*)Type_get_pointer(type, offset), 1, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_POINTER:{
            Type_receive_data_local((Type*)Type_get_pointer(type, offset), destination);
		    break;
		}
	}
}

void Type_receive_data_remote(Type* type, const int destination){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_INT:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_DOUBLE:
			MPI_Recv(type->currentPointer, type->currentLength, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_POINTER:{
		    int i;
		    for (i = type->currentLength - 1; i >= 0; --i){
    			Type_receive_data_remote(((Type*) (type->currentPointer)) + i, destination);
		    }
		    break;
		}
	}
}

void Type_receive_data_remote_offset(Type* type, const int destination, const long offset){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Recv((char*)Type_get_pointer(type, offset), 1, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_INT:
			MPI_Recv((int*)Type_get_pointer(type, offset), 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_DOUBLE:
			MPI_Recv((double*)Type_get_pointer(type, offset), 1, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			break;
		case TYPE_POINTER:{
            Type_receive_data_remote((Type*)Type_get_pointer(type, offset), destination);
		    break;
		}
	}
}

void Type_receive_meta_remote(Type* type, const int destination){
	MPI_Recv(&type->currentType, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&type->currentLength, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	switch(type->currentType){
	    case TYPE_CHAR:
		    type->currentPointer = malloc(sizeof(char) * type->currentLength); 
		    break;
	    case TYPE_INT:
    	    type->currentPointer = malloc(sizeof(int) * type->currentLength); 
		    break;
	    case TYPE_DOUBLE:
	        type->currentPointer = malloc(sizeof(double) * type->currentLength); 
		    break;
		case TYPE_POINTER:{
		    type->currentPointer = malloc(sizeof(Type) * type->currentLength);
		    int i;
		    for (i = type->currentLength - 1; i >= 0 ; --i){
			    Type_receive_meta_remote((Type*)Type_get_pointer(type, i), destination);
		    }
		    break;
	    }
	}
}

void Type_receive_meta_remote_type(Type* type, const int destination){
	MPI_Recv(&type->currentType, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Type_receive_meta_remote_offset(Type* type, const int destination, const long offset){
	switch(type->currentType){
		case TYPE_POINTER:{
	        Type_receive_meta_remote((Type*)Type_get_pointer(type, offset), destination);
	        break;
        }
    }
}


void Type_send_meta(Type* type, const int destination){
	MPI_Send(&type->currentType, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
	MPI_Send(&type->currentLength, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
	switch(type->currentType){
		case TYPE_POINTER:{
			int i;
			for (i = type->currentLength - 1; i >= 0; --i){
    			Type_send_meta((Type*)Type_get_pointer(type, i), destination);
			}
			break;
		}
	}
}

void Type_send_meta_type(Type* type, const int destination){
	MPI_Send(&type->currentType, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
}

void Type_send_meta_offset(Type* type, const int destination, const long offset){
	switch(type->currentType){
		case TYPE_POINTER:{
			Type_send_meta((Type*)Type_get_pointer(type, offset), destination);
			break;
		}
	}
}

void Type_send_data(Type* type, const int destination){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Send(((char*)(type->currentPointer)), type->currentLength, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_INT:
			MPI_Send(((int*)(type->currentPointer)), type->currentLength, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_DOUBLE:
			MPI_Send(((double*)(type->currentPointer)), type->currentLength, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_POINTER:
		{
		    int i;
			for (i = type->currentLength - 1; i >= 0; --i){
    			Type_send_data((Type*)Type_get_pointer(type, i), destination);
			}
			break;
		}
	}
}

void Type_send_data_offset(Type* type, const int destination, const long offset){
	switch(type->currentType){
		case TYPE_CHAR:
			MPI_Send((char*)Type_get_pointer(type, offset), 1, MPI_CHAR, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_INT:
			MPI_Send((int*)Type_get_pointer(type, offset), 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_DOUBLE:
			MPI_Send((double*)Type_get_pointer(type, offset), 1, MPI_DOUBLE, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			break;
		case TYPE_POINTER:
			Type_send_data((Type*)Type_get_pointer(type, offset), destination);
			break;
	}
}



void Type_delete(Type* type){
    Type_print(type);
    switch (type->currentType){
        case TYPE_POINTER:
        {
            int i;
			for (i = type->currentLength - 1; i >= 0; --i){
			    elogf("Ln:%d\n", __LINE__);
    			Type_delete((Type*)Type_get_pointer(type, i));
    			elogf("Ln:%d\n", __LINE__);
			}
			break;
		}
    }    
    free(type->currentPointer);
}







void add_two_arrays(const int result_global_index, const int result_local_index, Datapack* datapack){
	int* c = (int*) Type_get_pointer(datapack->typeList + 2, result_local_index);
	
	//const int len_c = datapack->typeList[2].currentLength;
	//const int len_a = datapack->typeList[0].currentLength;
	//const int len_b = datapack->typeList[1].currentLength;
	elogf("Ln:%d\n", __LINE__);
	int* a = (int*) Type_get_pointer(datapack->typeList + 0, result_local_index);
	elogf("Ln:%d\n", __LINE__);
	int* b = (int*) Type_get_pointer((Type*)Type_get_pointer(datapack->typeList + 1, result_local_index), 0);
	elogf("Ln:%d %p\n", __LINE__, b);
	//int i;
	//for (i = len_c; i > 0; --i){
	//	c[i] = a[i] + b[i];
	//}
	
	*c = *a + *b;
	printf("%d + %d = %d\n", *a, *b, *c);
}	

void lddot(const int result_global_index, const int result_local_index, Datapack* datapack){
	char * tra_a = (char*) Type_get_pointer(datapack->typeList + 0, result_local_index);
	char * tra_b = (char*) Type_get_pointer(datapack->typeList + 1, result_local_index);
	int * m = (int*) Type_get_pointer(datapack->typeList + 2, result_local_index);
	int * n = (int*) Type_get_pointer(datapack->typeList + 3, result_local_index);
	int * k = (int*) Type_get_pointer(datapack->typeList + 4, result_local_index);
	double * a_arr = ((Type*)Type_get_pointer(datapack->typeList + 5, result_local_index))->currentPointer;
	double * b_arr = ((Type*)Type_get_pointer(datapack->typeList + 6, result_local_index))->currentPointer;
	double * c_arr = ((Type*)Type_get_pointer(datapack->typeList + 7, result_local_index))->currentPointer;
	int i;
	NP_MATRIX a, b, c;
	NP_MATRIX *ptr2c = &c;
	wrap_np_matrix(&a, a_arr, *m, *k, *tra_a);
	wrap_np_matrix(&b, b_arr, *k, *n, *tra_b);
	wrap_np_matrix(&c, c_arr, *m, *n, 'N');
	ddot_np_matrix(&ptr2c, &a, &b);
}
static unsigned long function_table[] = {0, (unsigned long) add_two_arrays, (unsigned long) lddot};

void *local_worker(void *data){
    Workpack *workpack = (Workpack*) data;
    Controlpack *controlpack = (Controlpack*) workpack->controlpack;
    Datapack *datapack = (Datapack*) workpack->datapack;
	void (*func)(const int, const int, Datapack*) = (void (*)(const int, const int, Datapack*))(function_table[datapack->actualFunction]);
	
    int global_index_local;
    int local_indices[CGRAPE_SIZE] = {0};
    int local_count = 0;
    elogf("Ln:%d\n", __LINE__); 
    while (TRUE){
        sem_wait(&controlpack->counter_lock);
        while (local_count < CGRAPE_SIZE){
            
            while (controlpack->current_work_index < datapack->totalTaskNumber && datapack->isCompleted[controlpack->current_work_index] == 'Y'){
                ++controlpack->current_work_index;
            }
                
            if (controlpack->current_work_index >= datapack->totalTaskNumber){
                //sem_post(&controlpack->counter_lock);
                break;
            } else {
                local_indices[local_count++] = controlpack->current_work_index++;
            }
        }
        sem_post(&controlpack->counter_lock);
        while (local_count > 0){
            global_index_local = local_indices[--local_count];
            //Do actual work
			func(global_index_local, global_index_local, datapack);
        }       
        if (controlpack->current_work_index >= datapack->totalTaskNumber){
            pthread_exit(NULL); 
        }        
    }
} 

void *remote_worker(void *data){
    Workpack *workpack = (Workpack*) data;
    Controlpack *controlpack = (Controlpack*) workpack->controlpack;
    Datapack *datapack = (Datapack*) workpack->datapack;
	void (*func)(const int, const int, Datapack*) = (void (*)(const int, const int, Datapack*))(function_table[datapack->actualFunction]);
	
    int local_index;
	int global_index_local;
    while (TRUE){
        sem_wait(&controlpack->thread_lock);
        sem_wait(&controlpack->counter_lock);
        if (controlpack->current_work_count > 0){
            local_index = --controlpack->current_work_count;
			global_index_local = controlpack->remote_indices[local_index];
            sem_post(&controlpack->counter_lock);
            sem_wait(controlpack->resource_lock + local_index);
            //Call actual  function
            elogf("Ln:%d\n", __LINE__);
			func(global_index_local, local_index, datapack);
			
            sem_post(controlpack->product_lock + local_index); 
        } else if (controlpack->current_work_count < 0){
            sem_post(&controlpack->counter_lock);
            break;
        } else {
            sem_post(&controlpack->counter_lock);
        }
    }
    printf("Remote worker exiting\n");
    pthread_exit(NULL); 
}

void *driver(void *data){
    Workpack *workpack = (Workpack*) data;
    Controlpack *controlpack = (Controlpack*) workpack->controlpack;
    Datapack *datapack = (Datapack*) workpack->datapack;
	
    int global_index_local, i;
    int local_indices[WGRAPE_SIZE] = {0};
    int local_count, destination;
    int local_count_index;
    int total_count = 0;
	int hasShared = 0;
	
	sem_wait(&controlpack->counter_lock);
    destination = ++(controlpack->current_work_count);
    sem_post(&controlpack->counter_lock);
    while (TRUE){
        local_count = 0;
        sem_wait(&controlpack->counter_lock);
        
        while (local_count < WGRAPE_SIZE){
            
            while (controlpack->current_work_index < datapack->totalTaskNumber && datapack->isCompleted[controlpack->current_work_index] == 1){
                ++controlpack->current_work_index;
            }
                
            if (controlpack->current_work_index < datapack->totalTaskNumber){
                local_indices[local_count++] = controlpack->current_work_index++;
                ++total_count;
            } else{
                break;
            }
        }
        if (controlpack->current_work_index >= datapack->totalTaskNumber && local_count <= 0){
            sem_post(&controlpack->counter_lock);
            fprintf(stderr, "driver: destination=%d was distributed %d tasks\n", destination, total_count);
            MPI_Ssend(&local_count, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
            printf("Driver exiting\n");
            pthread_exit(NULL); 
        } 
        sem_post(&controlpack->counter_lock);
        MPI_Ssend(&local_count, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
        MPI_Send(local_indices, local_count, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
		
		if (hasShared == 0){
    		///Send datapack meta
			MPI_Send(&datapack->allEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			MPI_Send(&datapack->sharedEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			MPI_Send(&datapack->dependentEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			MPI_Send(&datapack->resultEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			MPI_Send(&datapack->totalTaskNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			
			if (datapack->sharedEntryNumber > 0){
				MPI_Send(datapack->sharedEntryIndices, datapack->sharedEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			}
			if (datapack->dependentEntryNumber > 0){ 
				MPI_Send(datapack->dependentEntryIndices, datapack->dependentEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			}
			
			if (datapack->resultEntryNumber > 0){
				MPI_Send(datapack->resultEntryIndices, datapack->resultEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD);
			}
			
			///Send shared meta and data
			for (i = datapack->sharedEntryNumber - 1; i >= 0; --i){
			    Type_send_meta(&datapack->typeList[datapack->sharedEntryIndices[i]], destination);
			    Type_send_data(&datapack->typeList[datapack->sharedEntryIndices[i]], destination);
			}
			
			///Send dependent meta type
			for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
			    Type_send_meta_type(&datapack->typeList[datapack->dependentEntryIndices[i]], destination);
			}
			
			///Send result meta type			
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
			        Type_send_meta_type(&datapack->typeList[datapack->resultEntryIndices[i]], destination);	
		        }
			}
			hasShared = 1;
		}
		elogf("Ln:%d\n", __LINE__);
		
		for (local_count_index = local_count - 1; local_count_index >= 0; --local_count_index){
			global_index_local = local_indices[local_count_index];
			for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
			    Type_send_meta_offset(&datapack->typeList[datapack->dependentEntryIndices[i]], destination, global_index_local);
			    Type_send_data_offset(&datapack->typeList[datapack->dependentEntryIndices[i]], destination, global_index_local);
			}
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
			        Type_send_meta_offset(&datapack->typeList[datapack->resultEntryIndices[i]], destination, global_index_local);
		        }
			}
		}
		
		elogf("Ln:%d\n", __LINE__);
		for (local_count_index = local_count - 1; local_count_index >= 0; --local_count_index){
			global_index_local = local_indices[local_count_index];
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    Type_receive_data_local_offset(&datapack->typeList[datapack->resultEntryIndices[i]], destination, global_index_local);
		    }
		}
		elogf("Ln:%d\n", __LINE__);
    }
}  


void *receiver(void *data){
	Workpack *workpack = (Workpack*) data;
    Controlpack *controlpack = (Controlpack*) workpack->controlpack;
    
    Datapack *datapack = (Datapack*) workpack->datapack;
	int local_count = 0;
    int destination = 0; // to root
	int hasShared = 0;
	
    int local_count_index, i, global_index_local;
    
    sem_wait(&controlpack->counter_lock);
    MPI_Recv(&controlpack->current_work_count, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    local_count = controlpack->current_work_count;
    
    printf("local count %d\n", local_count);
    sem_post(&controlpack->counter_lock);
	
    while (local_count > 0){
		MPI_Recv(&controlpack->remote_indices, local_count, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		if (hasShared == 0){
    		///Receive datapack meta
			MPI_Recv(&datapack->allEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&datapack->sharedEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&datapack->dependentEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&datapack->resultEntryNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&datapack->totalTaskNumber, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			datapack->sharedEntryIndices = malloc(sizeof(int) * datapack->sharedEntryNumber);
			datapack->dependentEntryIndices = malloc(sizeof(int) * datapack->dependentEntryNumber);
			datapack->resultEntryIndices = malloc(sizeof(int) * datapack->resultEntryNumber);
			datapack->typeList = malloc(sizeof(Type) * datapack->allEntryNumber);
			if (datapack->sharedEntryNumber > 0){
				MPI_Recv(datapack->sharedEntryIndices, datapack->sharedEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if (datapack->dependentEntryNumber > 0){ 
				MPI_Recv(datapack->dependentEntryIndices, datapack->dependentEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if (datapack->resultEntryNumber > 0){
				MPI_Recv(datapack->resultEntryIndices, datapack->resultEntryNumber, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			///End of receive datapack meta
			elogf("Ln:%d\n", __LINE__);
			///Receive shared meta and data
			for (i = datapack->sharedEntryNumber - 1; i >= 0; --i){
				Type_receive_meta_remote(&datapack->typeList[datapack->sharedEntryIndices[i]], destination);
				Type_receive_data_remote(&datapack->typeList[datapack->sharedEntryIndices[i]], destination);
			}
			
			///Receive dependent meta type
			for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
				elogf("Ln:%d\n", __LINE__);
			    Type_receive_meta_remote_type(&datapack->typeList[datapack->dependentEntryIndices[i]], destination);
			}
			
			///Receive result meta type
			elogf("Ln:%d\n", __LINE__);
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
			        Type_receive_meta_remote_type(&datapack->typeList[datapack->resultEntryIndices[i]], destination);	
		        }
			}
			hasShared = 1;
		}
		elogf("Ln:%d\n", __LINE__);
		for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
		    datapack->typeList[datapack->dependentEntryIndices[i]].currentLength = local_count;
		    datapack->typeList[datapack->dependentEntryIndices[i]].currentPointer = malloc(Type_sizeof(&datapack->typeList[datapack->dependentEntryIndices[i]]) * local_count);
		}
		
		elogf("Ln:%d\n", __LINE__);
		for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
		    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
		        datapack->typeList[datapack->resultEntryIndices[i]].currentLength = local_count;
		        datapack->typeList[datapack->resultEntryIndices[i]].currentPointer = malloc(Type_sizeof(&datapack->typeList[datapack->resultEntryIndices[i]]) * local_count);
	        }
		}

		elogf("Ln:%d\n", __LINE__);
		for (local_count_index = local_count - 1; local_count_index >= 0; --local_count_index){
			for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
		        Type_receive_meta_remote_offset(&datapack->typeList[datapack->dependentEntryIndices[i]], destination, local_count_index);
			    Type_receive_data_remote_offset(&datapack->typeList[datapack->dependentEntryIndices[i]], destination, local_count_index);
			}
			elogf("Ln:%d\n", __LINE__);
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
			        Type_receive_meta_remote_offset(&datapack->typeList[datapack->resultEntryIndices[i]], destination, local_count_index);
		        }
			}
            
			sem_post(controlpack->resource_lock + local_count_index);
			sem_post(&controlpack->thread_lock);
		}
		
		
		elogf("Ln:%d\n", __LINE__);
		for (local_count_index = local_count - 1; local_count_index >= 0; --local_count_index){
			sem_wait(controlpack->product_lock + local_count_index);
			elogf("Ln:%d\n", __LINE__);
			for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
			    Type_send_data_offset(&datapack->typeList[datapack->resultEntryIndices[i]], destination, local_count_index);
		    }
		}
		
		

		for (i = datapack->dependentEntryNumber - 1; i >= 0; --i){
            Type_delete(&datapack->typeList[datapack->dependentEntryIndices[i]]);
		}
		for (i = datapack->resultEntryNumber - 1; i >= 0; --i){
		    if (intbsearch(datapack->dependentEntryIndices, datapack->dependentEntryNumber,datapack->resultEntryIndices[i]) < 0){
                Type_delete(&datapack->typeList[datapack->resultEntryIndices[i]]);
	        }
		}
		
		elogf("Ln:%d\n", __LINE__);
		sem_wait(&controlpack->counter_lock);
		MPI_Recv(&controlpack->current_work_count, 1, MPI_INT, destination, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		local_count = controlpack->current_work_count;
		elogf("Ln:%d, local_count: %d\n", __LINE__, local_count);
		sem_post(&controlpack->counter_lock);
    }       
    sem_wait(&controlpack->counter_lock);
    controlpack->current_work_count = -1;
    sem_post(&controlpack->counter_lock);
    for (i = 0; i < NUM_OF_WTHREADS; i++){
        sem_post(&controlpack->thread_lock);
    }
	
	
	if (hasShared == 1){
		free(datapack->sharedEntryIndices);
		free(datapack->dependentEntryIndices);
		free(datapack->resultEntryIndices);
	    for (i = datapack->sharedEntryNumber - 1; i >= 0; --i){
		    Type_delete(&datapack->typeList[datapack->sharedEntryIndices[i]]);
	    }
		free(datapack->typeList);
	}
	

	printf("Receiver exiting\n");
    pthread_exit(NULL); 
}





void mpi_template_remote();
void mpi_template_local(Datapack *datapack){
	//Root:
	//Synchronizing point
	MPI_Bcast(&datapack->actualFunction, 1, MPI_INT, 0, MPI_COMM_WORLD);
	sg_signal = function_table[datapack->actualFunction];
	//Logger
	fprintf(stderr, "P%d: Inside mpi_template!\n", WORLD_RANK);
	double stime = omp_get_wtime(); 
	
	//Allocating control utilities
	Controlpack *controlpack  = malloc(sizeof(Controlpack));
	Workpack *workpack = malloc(sizeof(Workpack));
	workpack->controlpack = controlpack;
	workpack->datapack = datapack;
	sem_init(&controlpack->counter_lock, 0, 1);
	controlpack->current_work_index = 0;
	controlpack->current_work_count = 0; //Used as destinations

    //Allocating computing utilities
	pthread_t *driving_threads = malloc(sizeof(pthread_t) * NUM_OF_WTHREADS * (WORLD_SIZE - 1));
	pthread_t computing_threads[NUM_OF_CTHREADS];     
    elogf("Ln:%d\n", __LINE__);  
	int i;
	for (i = 0; i < NUM_OF_CTHREADS; ++i){
		pthread_create(computing_threads + i, NULL, local_worker, (void*) workpack);
	}
	if (NUM_OF_WTHREADS > 0){
		for (i = 0; i < WORLD_SIZE - 1; ++i){
			pthread_create(driving_threads + i, NULL, driver, workpack);
		}
	}
	for (i = 0; i < NUM_OF_CTHREADS; ++i){
		pthread_join(computing_threads[i], NULL);
	}
	fprintf(stderr, "mpi_template: %f seconds for cthreads to finish on root.\n", omp_get_wtime() - stime);
	if (NUM_OF_WTHREADS > 0){
		for (i = 0; i < WORLD_SIZE - 1; ++i){
			pthread_join(driving_threads[i], NULL);
		}
	}        
	
	//Finalizing
	free(driving_threads);
	free(workpack);
	free(controlpack);
	
	
    sem_destroy(&controlpack->counter_lock);
    fprintf(stderr, "mpi_template: %f seconds taken from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
}

void mpi_template_remote(){
	//Worker:  
	//Logger
    fprintf(stderr, "P%d: Inside mpi_template!\n", WORLD_RANK);
    double stime = omp_get_wtime();  
	
	//Creating workpack
	Workpack *workpack = malloc(sizeof(Workpack));
	
	//Allocating control utilities
	Controlpack *controlpack  = malloc(sizeof(Controlpack));
	
	workpack->controlpack = controlpack;
    sem_init(&controlpack->counter_lock, 0, 1);
	sem_init(&controlpack->thread_lock, 0, 0); //task lock : how many unfinished task in current comm
	int i;
	for (i = 0; i < WGRAPE_SIZE; ++i){
		sem_init(controlpack->resource_lock + i, 0, 0); //resource lock : if current task resource is ready
		sem_init(controlpack->product_lock + i, 0, 0); //product lock : if current product is ready
	}
    controlpack->current_work_index = 0;  
	controlpack->current_work_count = 0;  
	
	//Allocating computing utilities
	Datapack *datapack = malloc(sizeof(Datapack));
	workpack->datapack = datapack;
	datapack->actualFunction = sg_signal;
	pthread_t working_threads[NUM_OF_WTHREADS];
	pthread_t receiving_thread;
	pthread_create(&receiving_thread, NULL, receiver, (void*) workpack);
	for (i = 0; i < NUM_OF_WTHREADS; ++i){
		pthread_create(working_threads + i, NULL, remote_worker, (void*) workpack);
	}
	pthread_join(receiving_thread, NULL);
	for (i = 0; i < NUM_OF_WTHREADS; ++i){
		pthread_join(working_threads[i], NULL);
	}
	
	//Finalizing
	for (i = 0; i < WGRAPE_SIZE; ++i){
		sem_destroy(controlpack->resource_lock + i);
		sem_destroy(controlpack->product_lock + i);
	}
	sem_destroy(&controlpack->thread_lock);
    sem_destroy(&controlpack->counter_lock);
    fprintf(stderr, "mpi_template: %f seconds taken from processor %s, rank %d out of %d worlds.\n", omp_get_wtime() - stime, PROCESSOR_NAME, WORLD_RANK, WORLD_SIZE);
	
	
	free(datapack);
	free(controlpack);
	free(workpack);
}


void test_print_arr(Type *arr){
	int i;
	switch(arr->currentType){
		case TYPE_DOUBLE:
		    for (i = 0; i < arr->currentLength; ++i){
				fprintf(stderr, "%.3lf ", ((double*)(arr->currentPointer))[i]);
			}
			break;
		case TYPE_INT:
		    for (i = 0; i < arr->currentLength; ++i){
				fprintf(stderr, "%4d ", ((int*)(arr->currentPointer))[i]);
			}
			break;
	}
    fprintf(stderr, "\n---------------------------\n");
}

void mpi_final(){
	sg_signal = SIGNAL_EXIT;
	MPI_Bcast(&sg_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);	
	fprintf(stderr, "P%d: Finalized\n", WORLD_RANK);
	MPI_Finalize();		
	
}

void mpi_init(){
    
	int provided_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided_level);
	
    MPI_Comm_size(MPI_COMM_WORLD , &WORLD_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);

    MPI_Get_processor_name(PROCESSOR_NAME, &NAME_LEN);
    fprintf(stderr, "P%d: Initialized\n", WORLD_RANK);
	if (provided_level != MPI_THREAD_MULTIPLE){
		fprintf(stderr, "P%d: no multithreading support!\n", WORLD_RANK);
	}

    if (WORLD_RANK != 0){
        //In worker processes, block until root signals workers to continue
		//Synchronizing point
        MPI_Bcast(&sg_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        while (sg_signal != SIGNAL_EXIT){
            if (sg_signal == SIGNAL_SETONT){
                mpi_setONT(0);
            } else{
    			mpi_template_remote();
			}
			//MAGIC
			//void (*func)(void *);
			//func = (void (*)(void *))sg_signal;
            //(*func)(NULL);
			
            //After finishing a task, block again
			//Synchronizing point
            MPI_Bcast(&sg_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        fprintf(stderr, "P%d: Finalized\n", WORLD_RANK);

        MPI_Finalize();
    } else{
        //In root process, do nothing, just return
		fprintf(stdout, "NUM_OF_CTHREADS:\t%d\n", NUM_OF_CTHREADS);
		fprintf(stdout, "NUM_OF_WTHREADS:\t%d\n", NUM_OF_WTHREADS);
		fprintf(stdout, "CGRAPE_SIZE:\t%d\n", CGRAPE_SIZE);
		fprintf(stdout, "WGRAPE_SIZE:\t%d\n", WGRAPE_SIZE);
    }
}




int main(){	
    return 0;
}

