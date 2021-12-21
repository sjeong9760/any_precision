#ifndef _MODEL_PARAM_H
#define _MODEL_PARAM_H

#include <ap_int.h>

#include "model_shape.h"

#define AP_SIZE 32
#define GET_WEIGHT_ARRAY_LEN(x) ((x) / (AP_SIZE))
#define ROUND(x) ((x) <=0.5 ? 0 : 1)

typedef ap_uint<AP_SIZE> WEIGHT_T;

// Declare the array size of parameters
#define BB_0_ALPHA_BETA_LEN (bbShapes[0].bn_alpha_shape)
#define BB_0_CONV_BN_ALPHA_BETA_LEN (bbShapes[0].conv_bn_alpha_shape)
#define BB_0_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[0].conv_bn_weight_shape[0] * bbShapes[0].conv_bn_weight_shape[1] * \
    bbShapes[0].conv_bn_weight_shape[2] * bbShapes[0].conv_bn_weight_shape[3])
#define BB_0_CONV_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[0].conv_weight_shape[0] * bbShapes[0].conv_weight_shape[1] * \
    bbShapes[0].conv_weight_shape[2] * bbShapes[0].conv_weight_shape[3])


#define BB_1_ALPHA_BETA_LEN (bbShapes[1].bn_alpha_shape)
#define BB_1_SKIP_CONV_BN_ALPHA_BETA_LEN (bbShapes[1].skip_conv_bn_alpha_shape)
#define BB_1_SKIP_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] * \
    bbShapes[1].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3])
#define BB_1_CONV_BN_ALPHA_BETA_LEN (bbShapes[0].conv_bn_alpha_shape)
#define BB_1_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].conv_bn_weight_shape[0] * bbShapes[1].conv_bn_weight_shape[1] * \
    bbShapes[1].conv_bn_weight_shape[2] * bbShapes[1].conv_bn_weight_shape[3])
#define BB_1_CONV_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[1].conv_weight_shape[0] * bbShapes[1].conv_weight_shape[1] * \
    bbShapes[1].conv_weight_shape[2] * bbShapes[1].conv_weight_shape[3])


#define BB_2_ALPHA_BETA_LEN (bbShapes[2].bn_alpha_shape)
#define BB_2_CONV_BN_ALPHA_BETA_LEN (bbShapes[2].conv_bn_alpha_shape)
#define BB_2_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[2].conv_bn_weight_shape[0] * bbShapes[2].conv_bn_weight_shape[1] * \
    bbShapes[2].conv_bn_weight_shape[2] * bbShapes[2].conv_bn_weight_shape[3])
#define BB_2_CONV_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[2].conv_weight_shape[0] * bbShapes[2].conv_weight_shape[1] * \
    bbShapes[2].conv_weight_shape[2] * bbShapes[2].conv_weight_shape[3])


#define BB_3_ALPHA_BETA_LEN (bbShapes[3].bn_alpha_shape)
#define BB_3_SKIP_CONV_BN_ALPHA_BETA_LEN (bbShapes[3].skip_conv_bn_alpha_shape)
#define BB_3_SKIP_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].skip_conv_bn_weight_shape[0] * bbShapes[3].skip_conv_bn_weight_shape[1] * \
    bbShapes[3].skip_conv_bn_weight_shape[2] * bbShapes[3].skip_conv_bn_weight_shape[3])
#define BB_3_CONV_BN_ALPHA_BETA_LEN (bbShapes[0].conv_bn_alpha_shape)
#define BB_3_CONV_BN_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].conv_bn_weight_shape[0] * bbShapes[3].conv_bn_weight_shape[1] * \
    bbShapes[3].conv_bn_weight_shape[2] * bbShapes[3].conv_bn_weight_shape[3])
#define BB_3_CONV_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[3].conv_weight_shape[0] * bbShapes[3].conv_weight_shape[1] * \
    bbShapes[3].conv_weight_shape[2] * bbShapes[3].conv_weight_shape[3])


#define BB_4_ALPHA_BETA_LEN (bbShapes[4].bn_alpha_shape)
#define BB_4_CONV_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    bbShapes[4].conv_weight_shape[0] * bbShapes[4].conv_weight_shape[1] * \
    bbShapes[4].conv_weight_shape[2] * bbShapes[4].conv_weight_shape[3])


#define OUTER_ALPHA_BETA_LEN (otherShapes.bn_alpha_shape)
#define OUTER_FC_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    otherShapes.fc_weight_shape[0] * otherShapes.fc_weight_shape[1])
#define OUTER_FC_BIAS_LEN \
  GET_WEIGHT_ARRAY_LEN(otherShapes.fc_bias_shape)

float bb_0_bn_alpha_beta[BB_0_ALPHA_BETA_LEN];
float bb_0_conv_bn_alpha_beta[BB_0_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_0_conv_bn_weight[BB_0_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_0_conv_weight[BB_0_CONV_WEIGHT_LEN]

float bb_1_bn_alpha_beta[BB_1_ALPHA_BETA_LEN];
float bb_1_skip_conv_bn_alpha_beta[BB_1_SKIP_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_1_skip_conv_bn_weight[BB_1_SKIP_CONV_BN_WEIGHT_LEN];
float bb_1_conv_bn_alpha_beta[BB_1_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_1_conv_bn_weight[BB_1_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_1_conv_weight[BB_1_CONV_WEIGHT_LEN]

float bb_2_bn_alpha_beta[BB_2_ALPHA_BETA_LEN];
float bb_2_conv_bn_alpha_beta[BB_2_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_2_conv_bn_weight[BB_2_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_2_conv_weight[BB_2_CONV_WEIGHT_LEN]

float bb_3_bn_alpha_beta[BB_3_ALPHA_BETA_LEN];
float bb_3_skip_conv_bn_alpha_beta[BB_3_SKIP_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_3_skip_conv_bn_weight[BB_3_SKIP_CONV_BN_WEIGHT_LEN];
float bb_3_conv_bn_alpha_beta[BB_3_CONV_BN_ALPHA_BETA_LEN];
WEIGHT_T bb_3_conv_bn_weight[BB_3_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_3_conv_weight[BB_3_CONV_WEIGHT_LEN];

float bb_4_bn_alpha_beta[BB_4_ALPHA_BETA_LEN];
WEIGHT_T bb_4_conv_weight[BB_4_CONV_WEIGHT_LEN];

float outer_bn_alhpa_beta[OUTER_ALPHA_BETA_LEN];
WEIGHT_T fc_weight[OUTER_FC_WEIGHT_LEN];
float fc_bias[OUTER_FC_BIAS_LEN];

// Model parameters
typedef struct BasicBlockParams {
  float* bn_alpha_beta;

  float* skip_conv_bn_alpha_beta;
  WEIGHT_T* skip_conv_bn_weight;

  float* conv_bn_alpha_beta;
  WEIGHT_T* conv_bn_weight;

  WEIGHT_T* conv_weight;
} BasicBlockParams;

typedef struct OtherParams {
  float* bn_alpha_beta;

  WEIGHT_T* fc_weight;
  float* fc_bias;
  float fc_scale;
} OtherParams;

BasicBlockParams bbParams[NUM_BASIC_BLOCKS];
OtherParams otherParams;
// Basic Block 0
bbParams[0].bn_alpha_beta = bb_0_alpha_beta;
bbParams[0].conv_bn_alpha_beta = bb_0_conv_bn_alpha_beta;
bbParams[0].conv_bn_weight = bb_0_conv_bn_weight;
bbParams[0].conv_weight = bb_0_conv_weight;

// Basic Block 1
bbParams[1].bn_alpha_beta = bb_1_alpha_beta;
bbParams[1].skip_conv_bn_alpha_beta = bb_1_skip_conv_bn_alpha_beta;
bbParams[1].skip_conv_bn_weight = bb_1_skip_conv_bn_weight;
bbParams[1].conv_bn_alpha_beta = bb_1_conv_bn_alpha_beta;
bbParams[1].conv_bn_weight = bb_1_conv_bn_weight;
bbParams[1].conv_weight = bb_1_conv_weight;

// Basic Block 2
bbParams[2].bn_alpha_beta = bb_2_alpha_beta;
bbParams[2].conv_bn_alpha_beta = bb_2_conv_bn_alpha_beta;
bbParams[2].conv_bn_weight = bb_2_conv_bn_weight;
bbParams[2].conv_weight = bb_2_conv_weight;

// Basic Block 3
bbParams[3].bn_alpha_beta = bb_3_alpha_beta;
bbParams[3].skip_conv_bn_alpha_beta = bb_3_skip_conv_bn_alpha_beta;
bbParams[3].skip_conv_bn_weight = bb_3_skip_conv_bn_weight;
bbParams[3].conv_bn_alpha_beta = bb_3_conv_bn_alpha_beta;
bbParams[3].conv_bn_weight = bb_3_conv_bn_weight;
bbParams[3].conv_weight = bb_3_conv_weight;

// Basic Block 4
bbParams[4].bn_alpha_beta = bb_4_bn_alpha_beta;
bbParams[4].conv_weight = bb_4_conv_weight;

otherParams.bn_alpha_beta = outer_bn_alhpa_beta;
otherParams.fc_weight = outer_fc_weight;
otherParams.fc_bias = outer_fc_bias;
otherParams.fc_scale = 0.2489980161190033;

#endif


