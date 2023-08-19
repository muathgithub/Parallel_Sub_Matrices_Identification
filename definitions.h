#ifndef DEFINITIONS_H
#define  DEFINITIONS_H

#define TRUE 1
#define FALES 0
#define MATCHING_VALUE_TAG 0
#define PICTURES_NUM_TAG 1
#define OBJECTS_NUM_TAG 2
#define PICTURE_DATA_MESSAGE_TAG 3
#define OBJECT_DATA_MESSAGE_TAG 4
#define PICTURE_RESULT_TAG 5
#define RESULT_PICID_FOUND_NUM_SIZE 2
#define MASTER_RANK 0
#define WORKER_RANK 1
#define MAX_PICTURE_DIM 1000
#define PICTURE_ID_DIM_INT_SIZE 2
#define MAX_OBJECT_DIM 100
#define OBJECT_ID_DIM_INT_SIZE 2
#define MATCHING_VALUE 0
#define MATCHING_VALUE_LINE_NUM 0
#define PICTURES_NUM 1
#define PICTURES_NUM_LINE_NUM 1
#define OBJECTS_NUM 2
#define FOUND_OBJECT_INFO_SIZE 3
#define TO_FIND_OBJECTS_NUM 3
#define FOUND_OBJECTS_INFO_LEN (FOUND_OBJECT_INFO_SIZE * TO_FIND_OBJECTS_NUM)
#define NOT_MATCH -1
#define WARP_SIZE 32

typedef struct picture_struct {

  int picture_id, picture_dim, found_objects_num;
  int* picture_matrix;
  int found_objects_info [FOUND_OBJECTS_INFO_LEN];
  int * d_picture_matrix;
   
} picture_struct;

typedef struct object_struct {

  int object_id, object_dim;
  int* object_matrix;
  int* d_objects_matrix;

} object_struct;

#endif