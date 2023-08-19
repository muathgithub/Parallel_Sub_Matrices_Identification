
#ifndef CUDA_FUNCTIONS_H
#define  CUDA_FUNCTIONS_H
#include "definitions.h"

int compute_picture_on_GPU(picture_struct* curr_picture_struct,  object_struct* objects_structs_array, double matching_value, int picture_dim, int objects_num);
void update_found_objects_info(picture_struct* curr_picture_struct, int* h_result_array, int work_area_dim);
void free_objects_on_GPU(object_struct* objects_structs_array,  int objects_num);
void copy_object_to_GPU(object_struct* object_struct);

#endif