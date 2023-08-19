#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "definitions.h"
#include "cuda_functions.h"

__global__ void compute_picture_kernal(int* d_objects_matrix, int* d_picture_matrix, int object_id, int picture_dim, int object_dim, double matching_value, int* d_result_array, int work_area_dim);
__device__ double is_match(int* d_objects_matrix, int* d_picture_matrix, int picture_row_index, int picture_column_index, int picture_dim, int object_dim ,double matching_value);
void check_cuda_error(cudaError_t err);

// This function copies the object to the GPU using the multi stream concept (private stream for each call [thread])
void copy_object_to_GPU(object_struct* object_struct){

  cudaStream_t thread_cuda_stream;
  cudaError_t err = cudaSuccess;

  int block_dim = object_struct->object_dim;
  int object_size = pow(block_dim, 2);

  err = cudaStreamCreate(&thread_cuda_stream);
  check_cuda_error(err);

  // Allocating memory for the object matrinx in the GPU
  err = cudaMallocAsync(&object_struct->d_objects_matrix, sizeof(int) * object_size, thread_cuda_stream);
  check_cuda_error(err);
  
  // Copying the object matrix to the GPU
  err = cudaMemcpyAsync(object_struct->d_objects_matrix, object_struct->object_matrix, sizeof(int) * object_size, cudaMemcpyHostToDevice, thread_cuda_stream);
  check_cuda_error(err);

  // wating to the previous calls in the stream to be done
  err = cudaStreamSynchronize(thread_cuda_stream);
  check_cuda_error(err);

  err = cudaStreamDestroy(thread_cuda_stream); 
  check_cuda_error(err);
}


// This function called by each thread in the grid and each thread calculates if the object exists in the position
// according to the matching algorithm that given in the project file 
__device__ double is_match(int* d_objects_matrix, int* d_picture_matrix, int picture_row_index, int picture_column_index, int picture_dim, int object_dim ,double matching_value){
	
  int object_row_index , object_column_index;
  double matching_result = 0, curr_picture_cell, curr_object_cell;
  int curr_object_matrix_size = pow(object_dim, 2);


  // looping through the rows and columns to callculate if threre is a match according to the algorithm that given in the project file 
  for( object_row_index = 0; object_row_index < object_dim; object_row_index++){
    for( object_column_index = 0; object_column_index < object_dim; object_column_index++){

        curr_picture_cell = d_picture_matrix[picture_dim * (picture_row_index + object_row_index) + (picture_column_index + object_column_index)];

        curr_object_cell = d_objects_matrix[object_dim * object_row_index + object_column_index];

        matching_result += fabs((double)(((curr_picture_cell - curr_object_cell) / curr_picture_cell) / curr_object_matrix_size));  

        if(matching_result >= matching_value){

          return NOT_MATCH;
        }
      }
  }

  return matching_result;
}


// This function calls the function is_match() by a thread according to his position in the grid and the block
// for each possible position in the picture then it checks if there is a match according to the return value from is_match()
// and if true it fills the oject information and position in the result array
__global__ void compute_picture_kernal(int* d_objects_matrix, int* d_picture_matrix, int object_id, int picture_dim, int object_dim, double matching_value, int* d_result_array, int work_area_dim) {

  int picture_row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int picture_column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(picture_row_index < work_area_dim && picture_column_index <  work_area_dim){

    double result = is_match(d_objects_matrix, d_picture_matrix, picture_row_index, picture_column_index, picture_dim, object_dim, matching_value);

    if(result != NOT_MATCH){

      d_result_array[(picture_row_index * work_area_dim + picture_column_index) * FOUND_OBJECT_INFO_SIZE] = object_id;
      d_result_array[(picture_row_index * work_area_dim + picture_column_index) * FOUND_OBJECT_INFO_SIZE +1] = picture_row_index;
      d_result_array[(picture_row_index * work_area_dim + picture_column_index) * FOUND_OBJECT_INFO_SIZE + 2] = picture_column_index;
    }
  }
}


// This function checks for each possible/valid position in the picture if the curr object exists in it
// and it works in the multi stream concept (eeach thread that calls the function will create a private thread for itself)
int compute_picture_on_GPU(picture_struct* curr_picture_struct,  object_struct* objects_structs_array, double matching_value, int picture_dim, int objects_num){

  cudaStream_t thread_cuda_stream;
  cudaError_t err = cudaSuccess;
  int work_area_dim, result_array_size;
  int picture_size = pow(picture_dim, 2);
  int grid_dim;
  int* h_result_array = NULL;
  int* d_result_array = NULL;
  // The block size is static which is the max warp size = 32 so the blick size is 32*32
  dim3 block_dim3 = dim3(WARP_SIZE, WARP_SIZE);

  err = cudaStreamCreate(&thread_cuda_stream);
  check_cuda_error(err);

  // Allocating memory on the GPU for the picture matrix
  err = cudaMallocAsync(&curr_picture_struct->d_picture_matrix, picture_size * sizeof(int), thread_cuda_stream);
  check_cuda_error(err);

  // Copying the picture matrix to the GPu
  err = cudaMemcpyAsync(curr_picture_struct->d_picture_matrix,  curr_picture_struct->picture_matrix, picture_size * sizeof(int), cudaMemcpyHostToDevice, thread_cuda_stream);
  check_cuda_error(err);
  
  // Looping through the objects and for each object we check in parallel if it exists in all the posible positions in the picture
  for (int object_index = 0; object_index < objects_num; object_index++) {

    // the working area dimintion is picture_dim - curr_object.dim + 1 (so the object calculation dosent get out of the picture dimintions)
    work_area_dim = picture_dim - objects_structs_array[object_index].object_dim + 1;
    result_array_size = work_area_dim * work_area_dim * FOUND_OBJECT_INFO_SIZE;

    // Allocating memory on the host for the existing checks of the object in all the posible positions
    h_result_array = (int*) malloc(result_array_size * sizeof(int));
    if(h_result_array == NULL){ fprintf(stderr, "Host Failed To Malloc\n"); exit(EXIT_FAILURE); }

    // Allocating memory on the GPU for the existing checks of the object in all the posible positions
    err = cudaMallocAsync(&d_result_array, result_array_size * sizeof(int), thread_cuda_stream);
    check_cuda_error(err);
    
    // Initializing the result array with -1 so we can know if the object exists in the position or not
    err = cudaMemsetAsync(d_result_array, -1, result_array_size * sizeof(int), thread_cuda_stream);
    check_cuda_error(err);

    // Calculating the grid dimintion according to statis block dim which is the max warp size = 32 so the block size is 32*32
    grid_dim = work_area_dim % WARP_SIZE == 0 ? (work_area_dim / WARP_SIZE) : (work_area_dim / WARP_SIZE) + 1;

    dim3 grid_dim3 = dim3(grid_dim, grid_dim);

    // Calling the kernal to ckeck the exist of the object in all the posibble positions (using the thread private stream)
    compute_picture_kernal<<<grid_dim3, block_dim3, 0, thread_cuda_stream>>>(objects_structs_array[object_index].d_objects_matrix, curr_picture_struct->d_picture_matrix, objects_structs_array[object_index].object_id, picture_dim, objects_structs_array[object_index].object_dim, matching_value, d_result_array, work_area_dim);
    
    // copying the results array to the host
    err = cudaMemcpyAsync(h_result_array, d_result_array, result_array_size * sizeof(int), cudaMemcpyDeviceToHost, thread_cuda_stream);
    check_cuda_error(err);

    // wating till all the work on the current object have been done
    err = cudaStreamSynchronize(thread_cuda_stream);
    check_cuda_error(err);
    
    // updating the found objects info for the picture by calling the function update_found_objects_info()
    update_found_objects_info(curr_picture_struct, h_result_array, work_area_dim);
      
    free(h_result_array);
    
    err = cudaFreeAsync(d_result_array, thread_cuda_stream);
    check_cuda_error(err);

    // if three objects where foiund breake the loop
    if (curr_picture_struct->found_objects_num == TO_FIND_OBJECTS_NUM){ break; }   
  }

  err = cudaFreeAsync(curr_picture_struct->d_picture_matrix, thread_cuda_stream);
  check_cuda_error(err);

  // wating till the free finishes sueccesfully to destroy the stream after that
  err = cudaStreamSynchronize(thread_cuda_stream);
  check_cuda_error(err);

  err = cudaStreamDestroy(thread_cuda_stream);
  check_cuda_error(err);

  return 0;
}

// This function updates the found objects information in the picture struct after the the computation of that object
// have been done on the GPU h_result_array contains the object information fo each possible position in the picture 
void update_found_objects_info(picture_struct* curr_picture_struct, int* h_result_array, int work_area_dim){

  int result_objects_index = curr_picture_struct->found_objects_num;

  for(int i = 0; i < work_area_dim * work_area_dim; i++){

    if (h_result_array[i * FOUND_OBJECT_INFO_SIZE] != -1){

      curr_picture_struct->found_objects_info[result_objects_index * FOUND_OBJECT_INFO_SIZE] = h_result_array[i * FOUND_OBJECT_INFO_SIZE];
      curr_picture_struct->found_objects_info[result_objects_index * FOUND_OBJECT_INFO_SIZE + 1] = h_result_array[i * FOUND_OBJECT_INFO_SIZE + 1];
      curr_picture_struct->found_objects_info[result_objects_index * FOUND_OBJECT_INFO_SIZE + 2] = h_result_array[i * FOUND_OBJECT_INFO_SIZE + 2];

      curr_picture_struct->found_objects_num += 1;

      break;      
    }
  }
}

// This function frees the manually allocated memories for the objects on the GPU 
// using the multi streams concept
void free_objects_on_GPU(object_struct* objects_structs_array,  int objects_num){

  int object_index;
  cudaStream_t free_streams[objects_num];
  cudaError_t err = cudaSuccess;

  for (object_index = 0; object_index < objects_num; object_index++) {

     err = cudaStreamCreate(&free_streams[object_index]);
    check_cuda_error(err);

    err = cudaFreeAsync(objects_structs_array[object_index].d_objects_matrix, free_streams[object_index]);
    check_cuda_error(err);
  }

  // wating for all the works (frees) on all the streams to be done
  err = cudaDeviceSynchronize();
  check_cuda_error(err);

  for (object_index = 0; object_index < objects_num; object_index++) {

    err = cudaStreamDestroy(free_streams[object_index]); 
    check_cuda_error(err);
  }
}

// This function checks if threre is a cuda error if true it prints the cuda error string and exits/finishes the program
void check_cuda_error(cudaError_t err){

  if(err == cudaSuccess){
    return;
  }

  fprintf(stderr, "Cuda Error: %s)!\n", cudaGetErrorString(err)); 
  exit(EXIT_FAILURE); 
}
