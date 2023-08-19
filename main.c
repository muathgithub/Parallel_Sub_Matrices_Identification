#include <mpi.h>
#include "mpi_functions.h"
#include "cuda_functions.h"

int main(int argc, char * argv [] ) {

    // declaring the needed variaables
    MPI_Status status;
    int process_rank, pictures_num;

    // initializing the MPI enviroment
    MPI_Init(&argc, &argv);
    // getting the process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // This part is for the master
    if (process_rank == MASTER_RANK) {

        // variables for calculating the running time
        double time_1, time_2;
        time_1 = MPI_Wtime();

        char* input_file_path = argv[1];
        char* output_file_path = argv[2];

        // sending the main values to the worker (matching value, pictures number, objects number)
        send_value(input_file_path, MATCHING_VALUE_TAG);
        pictures_num = (int) send_value(input_file_path, PICTURES_NUM_TAG);
        send_value(input_file_path, OBJECTS_NUM_TAG);

        // reading and sending the objects information to the worker by calling the function read_and_send_objects()
        read_and_send_objects(input_file_path);
        // reading and sending the pictures information to the worker by calling the function read_and_send_pictures()
        read_and_send_pictures(input_file_path);

        // receving and writing the pictures results to the output file
        receive_and_write_results(output_file_path, pictures_num);

        // calculating and printing the running time
        time_2 = MPI_Wtime();
        printf("running_time = %e\n", (time_2 - time_1));
       
    } else if(process_rank == WORKER_RANK) {

        object_struct* objects_structs_array;
        picture_struct* pictures_structs_array;

        double matching_value;
        int objects_num;

        // receving the main values (matching value, pictures number, objects number) from the master
        matching_value = receive_value(MATCHING_VALUE_TAG);
        pictures_num = (int) receive_value(PICTURES_NUM_TAG);
        objects_num = (int) receive_value(OBJECTS_NUM_TAG);

        // starting/opening parallel region
        #pragma omp parallel
        {   
            // single region for calling the functions by one thread that creates tasks so the other threads start working on
            #pragma omp single
            {   
                // receving the objects from the master and copying them to the GPU in parallel using tasks and multi streams concepts
                objects_structs_array = receive_objects_with_GPU_copy(objects_num);
                // receving the pictures from the master and working on them on the GPU in parallel using tasks and multi streams concepts
                pictures_structs_array = receive_pictures_and_work(pictures_num, objects_structs_array, objects_num, matching_value); 
            }
        }

        // sending the pictures results to the master
        send_all_pictures_results(pictures_structs_array, pictures_num);

        // freeing the manually allocated memories in the GPU and in the Host
        free_objects_on_GPU(objects_structs_array, objects_num);
        free_pictures_structs_array(pictures_structs_array, pictures_num);
        free_objects_structs_array(objects_structs_array, objects_num);
    }

    // finalizing the Processes using MPI interface (terminate)
    MPI_Finalize();
    
    return 0;
}
