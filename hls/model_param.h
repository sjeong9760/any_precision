#ifndef _MODEL_PARAM_H
#define _MODEL_PARAM_H

#include <ap_int.h>

#include "model_shape.h"

#define AP_SIZE 32
#define GET_WEIGHT_ARRAY_LEN(x) ((x) / (AP_SIZE))

typedef ap_uint<AP_SIZE> WEIGHT_T;

// Declare the array size of parameters
#define BB_0_CONV_0_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[0].conv_0_shape[0] * bbShapes[0].conv_0_shape[1] * \
    bbShapes[0].conv_0_shape[2] * bbShapes[0].conv_0_shape[3])

#define BB_0_CONV_1_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[0].conv_1_shape[0] * bbShapes[0].conv_1_shape[1] * \
    bbShapes[0].conv_1_shape[2] * bbShapes[0].conv_1_shape[3])

#define BB_1_CONV_0_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].conv_0_shape[0] * bbShapes[1].conv_0_shape[1] * \
    bbShapes[1].conv_0_shape[2] * bbShapes[1].conv_0_shape[3])
#define BB_1_CONV_1_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].conv_1_shape[0] * bbShapes[1].conv_1_shape[1] * \
    bbShapes[1].conv_1_shape[2] * bbShapes[1].conv_1_shape[3])
#define BB_1_CONV_SKIP_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].conv_skip_shape[0] * bbShapes[1].conv_skip_shape[1] * \
    bbShapes[1].conv_skip_shape[2] * bbShapes[1].conv_skip_shape[3])

#define BB_2_CONV_0_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[2].conv_0_shape[0] * bbShapes[2].conv_0_shape[1] * \
    bbShapes[2].conv_0_shape[2] * bbShapes[2].conv_0_shape[3])
#define BB_2_CONV_1_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[2].conv_1_shape[0] * bbShapes[2].conv_1_shape[1] * \
    bbShapes[2].conv_1_shape[2] * bbShapes[2].conv_1_shape[3])

#define BB_3_CONV_0_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].conv_0_shape[0] * bbShapes[3].conv_0_shape[1] * \
    bbShapes[3].conv_0_shape[2] * bbShapes[3].conv_0_shape[3])
#define BB_3_CONV_1_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].conv_1_shape[0] * bbShapes[3].conv_1_shape[1] * \
    bbShapes[3].conv_1_shape[2] * bbShapes[3].conv_1_shape[3])
#define BB_3_CONV_SKIP_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].conv_skip_shape[0] * bbShapes[3].conv_skip_shape[1] * \
    bbShapes[3].conv_skip_shape[2] * bbShapes[3].conv_skip_shape[3])

#define BB_4_CONV_1_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[4].conv_1_shape[0] * bbShapes[4].conv_1_shape[1] * \
    bbShapes[4].conv_1_shape[2] * bbShapes[4].conv_1_shape[3])

#define FC_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    otherShapes.fc_weight_shape[0] * otherShapes.fc_weight_shape[1])


// CONV 0 parameter of Basic Block 0
WEIGHT_T bb_0_conv_0_weight[BB_0_CONV_0_WEIGHT_LEN];
// CONV 1 parameter of Basic Block 0
WEIGHT_T bb_0_conv_1_weight[BB_0_CONV_1_WEIGHT_LEN];

// CONV 0 parameter of Basic Block 1
WEIGHT_T bb_1_conv_0_weight[BB_1_CONV_0_WEIGHT_LEN];
// CONV 1 parameter of Basic Block 1
WEIGHT_T bb_1_conv_1_weight[BB_1_CONV_1_WEIGHT_LEN];
// SKIP CONV parameter of Basic Block 1
WEIGHT_T bb_1_conv_skip_weight[BB_1_CONV_SKIP_WEIGHT_LEN];

// CONV 0 parameter of Basic Block 2
WEIGHT_T bb_2_conv_0_weight[BB_2_CONV_0_WEIGHT_LEN];
// CONV 1 parameter of Basic Block 2
WEIGHT_T bb_2_conv_1_weight[BB_2_CONV_1_WEIGHT_LEN];

// CONV 0 parameter of Basic Block 3
WEIGHT_T bb_3_conv_0_weight[BB_3_CONV_0_WEIGHT_LEN];
// CONV 1 parameter of Basic Block 3
WEIGHT_T bb_3_conv_1_weight[BB_3_CONV_1_WEIGHT_LEN];
// SKIP CONV parameter of Basic Block 3
WEIGHT_T bb_3_conv_skip_weight[BB_3_CONV_SKIP_WEIGHT_LEN];

// CONV 1 parameter of Basic Block 4
WEIGHT_T bb_4_conv_1_weight[BB_4_CONV_1_WEIGHT_LEN];

// FC parameter
WEIGHT_T fc_weight[FC_WEIGHT_LEN];
float fc_bias[otherShapes.fc_bias_shape];

// Model parameters
typedef struct BasicBlockParams {
  WEIGHT_T* conv_0_weight;
  WEIGHT_T* conv_1_weight;
  WEIGHT_T* conv_skip_weight;
  float conv_0_scale;
  float conv_1_scale;
  float conv_skip_scale;
} BasicBlockParams;

typedef struct OtherParams {
  WEIGHT_T* fc_weight;
  float* fc_bias;
  float fc_scale;
} OtherParams;

BasicBlockParams bbParams[NUM_BASIC_BLOCKS];
OtherParams otherParams;
// Basic Block 0
bbParams[0].conv_0_weight = bb_0_conv_0_weight;
bbParams[0].conv_1_weight = bb_0_conv_1_weight;
bbParams[0].conv_0_scale = 0.2150881290435791;
bbParams[0].conv_1_scale = 0.22890882194042206;
// Basic Block 1
bbParams[1].conv_0_weight = bb_1_conv_0_weight;
bbParams[1].conv_1_weight = bb_1_conv_1_weight;
bbParams[1].conv_skip_weight = bb_1_conv_skip_weight;
bbParams[1].conv_0_scale = 0.2214481085538864;
bbParams[1].conv_1_scale = 0.19185993075370789;
bbParams[1].conv_skip_scale = 0.19089975953102112;
// Basic Block 2
bbParams[2].conv_0_weight = bb_2_conv_0_weight;
bbParams[2].conv_1_weight = bb_2_conv_1_weight;
bbParams[2].conv_0_scale = 0.20919275283813477;
bbParams[2].conv_1_scale = 0.2414044588804245;
// Basic Block 3
bbParams[3].conv_0_weight = bb_3_conv_0_weight;
bbParams[3].conv_1_weight = bb_3_conv_1_weight;
bbParams[3].conv_skip_weight = bb_3_conv_skip_weight;
bbParams[3].conv_0_scale = 0.20028334856033325;
bbParams[3].conv_1_scale = 0.19139564037322998;
bbParams[3].conv_skip_scale = 0.18395867943763733;
// Basic Block 4
bbParams[4].conv_1_weight = bb_4_conv_1_weight;
bbParams[4].conv_1_scale = 0.21254000067710876;

otherParams.fc_weight = fc_weight;
otherParams.fc_bias = fc_bias;
otherParams.fc_scale = 0.2489980161190033;

#endif


