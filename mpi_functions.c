#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"
#include "mpi_functions.h"
#include "cuda_functions.h"

// This function loops through all the pictures and it sends there results to the master
// using the send_picture_result() function that sends one picture result to the master
void send_all_pictures_results(picture_struct* pictures_structs_array, int pictures_num){

    int picture_index;
    for (picture_index = 0; picture_index < pictures_num; picture_index++){

        send_picture_result(&pictures_structs_array[picture_index]);
    }
}

// This function sends the needed value from the input file according to the given value_tag
// using the read_value() that reads the needed value from the file according to the line that it exists it
// int the case of getting the line number of the objects_num value it uses the get_objects_num_line_num() function
double send_value(char* input_file_path, int value_tag){

    double value;

    switch (value_tag){

        case MATCHING_VALUE_TAG:
            value = read_value(input_file_path, MATCHING_VALUE_LINE_NUM);
            MPI_Send(&value, 1,  MPI_DOUBLE, WORKER_RANK, MATCHING_VALUE_TAG, MPI_COMM_WORLD);
            break;
        
        case PICTURES_NUM_TAG:
            value = read_value(input_file_path, PICTURES_NUM_TAG);
            MPI_Send(&value, 1,  MPI_DOUBLE, WORKER_RANK, PICTURES_NUM_TAG, MPI_COMM_WORLD);
            break;

        case OBJECTS_NUM_TAG:
            value = read_value(input_file_path, get_objects_num_line_num(input_file_path));
            MPI_Send(&value, 1,  MPI_DOUBLE, WORKER_RANK, OBJECTS_NUM_TAG, MPI_COMM_WORLD);
            break;
    }

    return value;
}

// This function receives the needed value from the master according to the given value_tag 
double receive_value(int value_tag){

    double value;
    MPI_Status status;

    switch (value_tag){

        case MATCHING_VALUE_TAG:
            MPI_Recv(&value, 1, MPI_DOUBLE, MASTER_RANK, MATCHING_VALUE_TAG, MPI_COMM_WORLD, &status);
            break;
        
        case PICTURES_NUM_TAG:
            MPI_Recv(&value, 1, MPI_DOUBLE, MASTER_RANK, PICTURES_NUM_TAG, MPI_COMM_WORLD, &status);
            break;

        case OBJECTS_NUM_TAG:
            MPI_Recv(&value, 1, MPI_DOUBLE, MASTER_RANK, OBJECTS_NUM_TAG, MPI_COMM_WORLD, &status);
            break;
    
        default:
            return -1;
    }

    return value;
}

// This function reads a value from the input file according to the given value line number
double read_value(char* input_file_path, int value_line_num) {

    double value;
    int curr_line_num = 0;
    char* curr_line = NULL;
    size_t len = 0;
    size_t read;

    FILE* file = fopen (input_file_path, "r");

    if (!file) {
        perror(input_file_path);
        exit(EXIT_FAILURE);
    }

    // looping through the file lines till we get to the needed line then we read the value and return it 
    while ((read = getline(&curr_line, &len, file)) != -1){

        if(curr_line_num == value_line_num){

            value = strtod(curr_line, NULL);
            free(curr_line);
            fclose (file);
            return value;
        }

        curr_line_num += 1;
    }
    
    free(curr_line);
    fclose(file);
    return -1;    
}

// This function returns the line number of the line that contains the number of the objects in the file
int get_objects_num_line_num(char* input_file_path){

    int curr_line_num = 3, to_skip_lines = 3, curr_picture_dim = 0;
     // getting the pictures num for the purpose of skipping the pictures information lines 
    int pictures_num = (int) read_value(input_file_path, PICTURES_NUM_TAG);
    char* curr_line = NULL;
    size_t len = 0;
    size_t read;

    FILE* file = fopen (input_file_path, "r");

    if (!file) {
        perror(input_file_path);
        exit(EXIT_FAILURE);
    } 

    while ((read = getline(&curr_line, &len, file)) != -1){

        // skipping the first lines till we get to the first picture dimintion value
        if(to_skip_lines > 0){

            to_skip_lines -= 1;
            continue;

        // if the number of pictures == 0 that means that we skipped all the pictures information and we got to the objects number line
        // so we return the line number of that objects number value
        } else if (pictures_num == 0) {
            
            free(curr_line);
            fclose (file);
            return curr_line_num;
        } 

        curr_picture_dim = (int) strtol(curr_line, (char **)NULL, 10);
        pictures_num -= 1;
         // the + 1 is for getting to the next line that contains the next picture id
        to_skip_lines = pictures_num == 0 ? curr_picture_dim : curr_picture_dim + 1; 
        // the + 1 is for skipping the next picture id or to get to the next line that contains the objects num in case of pictures_num == 0
        curr_line_num += to_skip_lines + 1; 
    }

    free(curr_line);
    fclose (file);
    return -1;
}

// This function called by the master to start reading the pictures information and sending them to the worker
// it reads the picture information from the file and it sends them as one message (array of int)
// the message format is [picture_id (int), picture_dim(int), picture_matrix(multiple int)]
void read_and_send_pictures(char* input_file_path){

    int to_skip_lines = 2;
    int pictures_num = (int) read_value(input_file_path, PICTURES_NUM_LINE_NUM);
    int picture_data_message[(int)(PICTURE_ID_DIM_INT_SIZE + pow(MAX_PICTURE_DIM, 2))];
    char* curr_line = NULL;
    size_t len = 0;
    size_t read;

    FILE* file = fopen (input_file_path, "r");

    if (!file) { perror(input_file_path); exit(EXIT_FAILURE); }

    // skipping the uneeded lines from the input file
    while (to_skip_lines > 0){ getline(&curr_line, &len, file); to_skip_lines -= 1; } free(curr_line);

    // going through the file and sending each picture information in the explained above format
    int i, j, picture_matrix_size;
    for(i = 0; i < pictures_num; i++){

        fscanf(file, "%d", &(picture_data_message[0]));
        fscanf(file, "%d", &(picture_data_message[1]));

        picture_matrix_size = pow(picture_data_message[1], 2);

        for(j = 0; j < picture_matrix_size; j++){

            fscanf(file, "%d", &(picture_data_message[PICTURE_ID_DIM_INT_SIZE + j]));
        }

        MPI_Send(picture_data_message, (PICTURE_ID_DIM_INT_SIZE + picture_matrix_size),  MPI_INT, WORKER_RANK, PICTURE_DATA_MESSAGE_TAG, MPI_COMM_WORLD);
    }

    fclose(file);
}

// This function called by the worker it receives the pictures information messages from the master and it puts them in picture_struct
// this function called by a single thread in a parallel reagon from the main function and after receving each picture
// the single thread creates a task so the other thread can starts working on the picture by calling the function compute_picture_on_GPU()
// that works on private stream (the program works/uses on multi streams concept) with multi threads that work on OpenMP tasks
picture_struct* receive_pictures_and_work(int pictures_num, object_struct* objects_structs_array, int objects_num, double matching_value){
    
    MPI_Status status;
    int max_message_len = PICTURE_ID_DIM_INT_SIZE + pow(MAX_PICTURE_DIM, 2);
    int picture_data_message[max_message_len];
    picture_struct* pictures_structs_array = (picture_struct*) malloc(pictures_num * sizeof(picture_struct));
   
    int i, picture_matrix_size;
    for(i = 0; i < pictures_num; i++){
        MPI_Recv(picture_data_message, max_message_len, MPI_INT, MASTER_RANK, PICTURE_DATA_MESSAGE_TAG, MPI_COMM_WORLD, &status);

        pictures_structs_array[i].picture_id = picture_data_message[0];
        pictures_structs_array[i].picture_dim = picture_data_message[1];
        pictures_structs_array[i].found_objects_num = 0;
        picture_matrix_size = pow(picture_data_message[1], 2);

        pictures_structs_array[i].picture_matrix = (int*) malloc(picture_matrix_size * sizeof(int));
        memcpy(pictures_structs_array[i].picture_matrix, &(picture_data_message[PICTURE_ID_DIM_INT_SIZE]), picture_matrix_size * sizeof(int));

        #pragma omp task
        {
            compute_picture_on_GPU(&pictures_structs_array[i], objects_structs_array, matching_value, pictures_structs_array[i].picture_dim , objects_num);
        }
    }
    // wating for all the child tasks to be done
    #pragma omp taskwait

    return pictures_structs_array;
}

// This function called by the master to start reading the objects information and sending them to the worker
// it reads the object information from the file and it sends them as one message (array of int)
// the message format is [object_id (int), object_dim(int), object_matrix(multiple int)]
void read_and_send_objects(char* input_file_path){

    int objects_num_line_num = get_objects_num_line_num(input_file_path);
    int to_skip_lines = objects_num_line_num + 1;
    int objects_num = (int) read_value(input_file_path, objects_num_line_num);
    int object_data_message[(int)(OBJECT_ID_DIM_INT_SIZE + pow(MAX_OBJECT_DIM, 2))];
    char* curr_line = NULL;
    size_t len = 0;
    size_t read;

    FILE* file = fopen (input_file_path, "r");

    if (!file) { perror(input_file_path); exit(EXIT_FAILURE); }

    // skipping the uneeded lines from the input file
    while (to_skip_lines > 0){ getline(&curr_line, &len, file); to_skip_lines -= 1; } free(curr_line);

    // going through the file and sending each object information in the explained above format
    int i, j, object_matrix_size;
    for(i = 0; i < objects_num; i++){

        fscanf(file, "%d", &(object_data_message[0]));
        fscanf(file, "%d", &(object_data_message[1]));

        object_matrix_size = pow(object_data_message[1], 2);

        for(j = 0; j < object_matrix_size; j++){

            fscanf(file, "%d", &(object_data_message[OBJECT_ID_DIM_INT_SIZE + j]));
        }

        MPI_Send(object_data_message, (OBJECT_ID_DIM_INT_SIZE + object_matrix_size),  MPI_INT, WORKER_RANK, OBJECT_DATA_MESSAGE_TAG, MPI_COMM_WORLD);
    }

    fclose(file);
}

// This function called by the worker it receives the objects information messages from the master and it puts them in object_struct
// this function called by a single thread in a parallel reagon from the main function and after receving each object
// the single thread creates a task so the other thread can starts copying the object to the GPU by calling the function copy_object_to_GPU()
// that copies the object to the GPU using private stream (the program works/uses on multi streams concept) with multi threads that work on OpenMP tasks
object_struct* receive_objects_with_GPU_copy(int objects_num){

    MPI_Status status;
    int max_message_len = OBJECT_ID_DIM_INT_SIZE + pow(MAX_OBJECT_DIM, 2);
    int object_data_message[max_message_len];
    object_struct* objects_structs_array = (object_struct*) malloc(objects_num * sizeof(object_struct));
   
    int i, object_matrix_size;
    for(i = 0; i < objects_num; i++){ 

        MPI_Recv(object_data_message, max_message_len, MPI_INT, MASTER_RANK, OBJECT_DATA_MESSAGE_TAG, MPI_COMM_WORLD, &status);

        objects_structs_array[i].object_id = object_data_message[0];
        objects_structs_array[i].object_dim = object_data_message[1];
        object_matrix_size = pow(object_data_message[1], 2);

        objects_structs_array[i].object_matrix = (int*) malloc(object_matrix_size * sizeof(int));
        memcpy(objects_structs_array[i].object_matrix, &(object_data_message[OBJECT_ID_DIM_INT_SIZE]), object_matrix_size * sizeof(int));

        #pragma omp task
        {
            copy_object_to_GPU(&objects_structs_array[i]);
        }
    }

    // wating for all the child tasks to be done
    #pragma omp taskwait

    return objects_structs_array;
}

// This function called by the worker to send the picture result to the mater as one message (array of int)
// the message format [picture_id (int), foun_objects_num * (FOUND_OBJECT_INFO_SIZE = 3 [object_id, x_position, y_position])]
void send_picture_result(picture_struct* curr_picture_struct){

    int found_objects_num = curr_picture_struct->found_objects_num;
    int result_len = RESULT_PICID_FOUND_NUM_SIZE + (found_objects_num * FOUND_OBJECT_INFO_SIZE);
    int picture_result_message[result_len]; 

    picture_result_message[0] = curr_picture_struct->picture_id;
    picture_result_message[1] = found_objects_num;
    
    if(found_objects_num > 0){
        memcpy(&(picture_result_message[2]), curr_picture_struct->found_objects_info, found_objects_num *  FOUND_OBJECT_INFO_SIZE * sizeof(int));
    }

    MPI_Send(picture_result_message, result_len, MPI_INT, MASTER_RANK, PICTURE_RESULT_TAG, MPI_COMM_WORLD);
}

// This function called by the master to recive the pictures results from the worker and to write them to the outputfile
// the picture resutl message format is [picture_id (int), foun_objects_num * (FOUND_OBJECT_INFO_SIZE = 3 [object_id, x_position, y_position])]
void receive_and_write_results(char* output_file_path, int pictures_num){

    MPI_Status status;
    int max_message_len = RESULT_PICID_FOUND_NUM_SIZE + FOUND_OBJECTS_INFO_LEN;
    int picture_result_message [max_message_len];
    char output_line[300];
    int* found_objects_info;

    FILE *file = fopen(output_file_path, "w");

    if (!file) { perror(output_file_path); exit(EXIT_FAILURE); }

    int i, j;
    for(i = 0; i < pictures_num; i++){

        MPI_Recv(picture_result_message, max_message_len, MPI_INT, WORKER_RANK, PICTURE_RESULT_TAG, MPI_COMM_WORLD, &status);

        if(picture_result_message[1] == TO_FIND_OBJECTS_NUM){

            char to_cat_string[250];
            found_objects_info = &(picture_result_message[2]);

            sprintf(output_line, "Picture %d: found Objects:", picture_result_message[0]);

            for(j = 0; j < FOUND_OBJECTS_INFO_LEN; j += FOUND_OBJECT_INFO_SIZE){

                sprintf(to_cat_string, " %d Position(%d,%d)", found_objects_info[j], found_objects_info[j + 1], found_objects_info[j + 2]);

                if(j < FOUND_OBJECTS_INFO_LEN - FOUND_OBJECT_INFO_SIZE) { strcat(to_cat_string, " ;"); } else { strcat(to_cat_string, "\n"); }

                strcat(output_line, to_cat_string);
            }

            fprintf(file, "%s", output_line);

        } else {

            sprintf(output_line, "Picture %d: No three different Objects were found\n", picture_result_message[0]);
            fprintf(file, "%s", output_line);
        }
    }

    fclose(file);  
}

// This function called by the worker after finishing the work on the pictures to free all the manually allocated
// memories for saving the pictures structs and matrixes
void free_pictures_structs_array(picture_struct* pictures_structs_array, int pictures_num){

    int picture_index;
    for(picture_index = 0; picture_index < pictures_num; picture_index++){

        free(pictures_structs_array[picture_index].picture_matrix);
    }

    free(pictures_structs_array);
}

// This function called by the worker after finishing the work on the pictures to free all the manually allocated
// memories for saving the objects structs and matrixes
void free_objects_structs_array(object_struct* objects_structs_array, int objects_num){

    int object_index;
    for(object_index = 0; object_index < objects_num; object_index++){

        free(objects_structs_array[object_index].object_matrix);
    }

    free(objects_structs_array);
}
