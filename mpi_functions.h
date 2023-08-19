
#ifndef MPI_FUNCTIONS_H
#define  MPI_FUNCTIONS_H
#include "definitions.h"

double send_value(char* input_file_path, int value_tag);
double receive_value(int value_tag);
double read_value(char* input_file_path, int value_line_num);
int get_objects_num_line_num(char* input_file_path);
void read_and_send_pictures(char* input_file_path);
picture_struct* receive_pictures_and_work(int pictures_num, object_struct* objects_structs_array, int objects_num, double matching_value);
void read_and_send_objects(char* input_file_path);
object_struct* receive_objects_with_GPU_copy(int objects_num);
void free_pictures_structs_array(picture_struct* pictures_structs_array, int pictures_num);
void free_objects_structs_array(object_struct* objects_structs_array, int objects_num);
void send_all_pictures_results(picture_struct* pictures_structs_array, int pictures_num);
void send_picture_result(picture_struct* curr_picture_struct);
void receive_and_write_results(char* outtput_file_path, int pictures_num);

#endif