#ifndef _MODEL_PARAM_H
#define _MODEL_PARAM_H

#include <ap_int.h>

#include "model_shape.h"

#define AP_SIZE 32
#define GET_WEIGHT_ARRAY_LEN(x) ((x) / (AP_SIZE))

// Declare the array size of parameters
const uint16_t bb_0_conv_0_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[0].conv_0_shape[0] * bbShapes[0].conv_0_shape[1] * 
    bbShapes[0].conv_0_shape[2] * bbShapes[0].conv_0_shape[3]);
const uint16_t bb_0_conv_1_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[0].conv_1_shape[0] * bbShapes[0].conv_1_shape[1] * 
    bbShapes[0].conv_1_shape[2] * bbShapes[0].conv_1_shape[3]);

const uint16_t bb_1_conv_0_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[1].conv_0_shape[0] * bbShapes[1].conv_0_shape[1] * 
    bbShapes[1].conv_0_shape[2] * bbShapes[1].conv_0_shape[3]);
const uint16_t bb_1_conv_1_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[1].conv_1_shape[0] * bbShapes[1].conv_1_shape[1] * 
    bbShapes[1].conv_1_shape[2] * bbShapes[1].conv_1_shape[3]);
const uint16_t bb_1_conv_skip_weight_len = 
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[1].conv_skip_shape[0] * bbShapes[1].conv_skip_shape[1] * 
    bbShapes[1].conv_skip_shape[2] * bbShapes[1].conv_skip_shape[3]);

const uint16_t bb_2_conv_0_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[2].conv_0_shape[0] * bbShapes[2].conv_0_shape[1] * 
    bbShapes[2].conv_0_shape[2] * bbShapes[2].conv_0_shape[3]);
const uint16_t bb_2_conv_1_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[2].conv_1_shape[0] * bbShapes[2].conv_1_shape[1] * 
    bbShapes[2].conv_1_shape[2] * bbShapes[2].conv_1_shape[3]);

const uint16_t bb_3_conv_0_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[3].conv_0_shape[0] * bbShapes[3].conv_0_shape[1] * 
    bbShapes[3].conv_0_shape[2] * bbShapes[3].conv_0_shape[3]);
const uint16_t bb_3_conv_1_weight_len =
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[3].conv_1_shape[0] * bbShapes[3].conv_1_shape[1] * 
    bbShapes[3].conv_1_shape[2] * bbShapes[3].conv_1_shape[3]);
const uint16_t bb_3_conv_skip_weight_len = 
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[3].conv_skip_shape[0] * bbShapes[3].conv_skip_shape[1] * 
    bbShapes[3].conv_skip_shape[2] * bbShapes[3].conv_skip_shape[3]);

const uint16_t bb_4_conv_1_weight_len = 
  GET_WEIGHT_ARRAY_LEN(
    bbShapes[4].conv_1_shape[0] * bbShapes[4].conv_1_shape[1] * 
    bbShapes[4].conv_1_shape[2] * bbShapes[4].conv_1_shape[3]);

const uint16_t fc_weight_len = 
  GET_WEIGHT_ARRAY_LEN(
    otherShapes.fc_weight_shape[0] * otherShapes.fc_weight_shape[1]);


// CONV 0 parameter of Basic Block 0
ap_int<AP_SIZE> bb_0_conv_0_weight[bb_0_conv_0_weight_len];
// CONV 1 parameter of Basic Block 0
ap_int<AP_SIZE> bb_0_conv_1_weight[bb_0_conv_1_weight_len];

// CONV 0 parameter of Basic Block 1
ap_int<AP_SIZE> bb_1_conv_0_weight[bb_1_conv_0_weight_len];
// CONV 1 parameter of Basic Block 1
ap_int<AP_SIZE> bb_1_conv_1_weight[bb_1_conv_1_weight_len];
// SKIP CONV parameter of Basic Block 1
ap_int<AP_SIZE> bb_1_conv_skip_weight[bb_1_conv_skip_weight_len];

// CONV 0 parameter of Basic Block 2
ap_int<AP_SIZE> bb_2_conv_0_weight[bb_2_conv_0_weight_len];
// CONV 1 parameter of Basic Block 2
ap_int<AP_SIZE> bb_2_conv_1_weight[bb_2_conv_1_weight_len];

// CONV 0 parameter of Basic Block 3
ap_int<AP_SIZE> bb_3_conv_0_weight[bb_3_conv_0_weight_len];
// CONV 1 parameter of Basic Block 3
ap_int<AP_SIZE> bb_3_conv_1_weight[bb_3_conv_1_weight_len];
// SKIP CONV parameter of Basic Block 3
ap_int<AP_SIZE> bb_3_conv_skip_weight[bb_3_conv_skip_weight_len];

// CONV 1 parameter of Basic Block 4
ap_int<AP_SIZE> bb_4_conv_1_weight[bb_4_conv_1_weight_len];

// FC parameter
ap_int<AP_SIZE> fc_weight[fc_weight_len];
float fc_bias[otherShapes.fc_bias_shape];

// Model parameters
typedef struct BasicBlockParams {
  ap_int<AP_SIZE>* conv_0_weight;
  ap_int<AP_SIZE>* conv_1_weight;
  ap_int<AP_SIZE>* conv_skip_weight;
  float conv_0_scale;
  float conv_1_scale;
  float conv_skip_scale;
} BasicBlockParams;

typedef struct OtherParams {
  ap_int<AP_SIZE>* fc_weight;
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


