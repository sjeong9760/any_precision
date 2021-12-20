#ifndef _MODEL_PARAM_H
#define _MODEL_PARAM_H

#include <ap_int.h>

#include "model_shape.h"

// CONV 0 parameter of Basic Block 0
ap_int<1> bb_0_conv_0_weight[
  bbShapes[0].conv_0_shape[0] * bbShapes[0].conv_0_shape[1] * 
  bbShapes[0].conv_0_shape[2] * bbShapes[0].conv_0_shape[3]
];
// CONV 1 parameter of Basic Block 0
ap_int<1> bb_0_conv_1_weight[
  bbShapes[0].conv_1_shape[0] * bbShapes[0].conv_1_shape[1] * 
  bbShapes[0].conv_1_shape[2] * bbShapes[0].conv_1_shape[3]
];

// CONV 0 parameter of Basic Block 1
ap_int<1> bb_1_conv_0_weight[
  bbShapes[1].conv_0_shape[0] * bbShapes[1].conv_0_shape[1] * 
  bbShapes[1].conv_0_shape[2] * bbShapes[1].conv_0_shape[3]
];
// CONV 1 parameter of Basic Block 1
ap_int<1> bb_1_conv_1_weight[
  bbShapes[1].conv_1_shape[0] * bbShapes[1].conv_1_shape[1] * 
  bbShapes[1].conv_1_shape[2] * bbShapes[1].conv_1_shape[3]
];
// SKIP CONV parameter of Basic Block 1
ap_int<1> bb_1_conv_skip_weight[
  bbShapes[1].conv_skip_shape[0] * bbShapes[1].conv_skip_shape[1] * 
  bbShapes[1].conv_skip_shape[2] * bbShapes[1].conv_skip_shape[3]
];

// CONV 0 parameter of Basic Block 2
ap_int<1> bb_0_conv_0_weight[
  bbShapes[2].conv_0_shape[0] * bbShapes[0].conv_0_shape[1] * 
  bbShapes[2].conv_0_shape[2] * bbShapes[0].conv_0_shape[3]
];
// CONV 1 parameter of Basic Block 2
ap_int<1> bb_0_conv_1_weight[
  bbShapes[2].conv_1_shape[0] * bbShapes[0].conv_1_shape[1] * 
  bbShapes[2].conv_1_shape[2] * bbShapes[0].conv_1_shape[3]
];

// CONV 0 parameter of Basic Block 3
ap_int<1> bb_1_conv_0_weight[
  bbShapes[3].conv_0_shape[0] * bbShapes[3].conv_0_shape[1] * 
  bbShapes[3].conv_0_shape[2] * bbShapes[3].conv_0_shape[3]
];
// CONV 1 parameter of Basic Block 3
ap_int<1> bb_1_conv_1_weight[
  bbShapes[3].conv_1_shape[0] * bbShapes[3].conv_1_shape[1] * 
  bbShapes[3].conv_1_shape[2] * bbShapes[3].conv_1_shape[3]
];
// SKIP CONV parameter of Basic Block 3
ap_int<1> bb_1_conv_skip_weight[
  bbShapes[3].conv_skip_shape[0] * bbShapes[3].conv_skip_shape[1] * 
  bbShapes[3].conv_skip_shape[2] * bbShapes[3].conv_skip_shape[3]
];

// CONV 1 parameter of Basic Block 4
ap_int<1> bb_4_conv_1_weight[
  bbShapes[4].conv_1_shape[0] * bbShapes[4].conv_1_shape[1] * 
  bbShapes[4].conv_1_shape[2] * bbShapes[4].conv_1_shape[3]
];

// Model parameters
typedef struct BasicBlockParams {
  ap_int<1>* conv_0_weight;
  ap_int<1>* conv_1_weight;
  ap_int<1>* conv_skip_weight;
} BasicBlockParams;

typedef struct OtherParams {
  ap_int<1>* fc_weight;
  ap_int<1>* fc_bias;
} OtherParams;

BasicBlockParams bbParams[NUM_BASIC_BLOCKS];
// Basic Block 0
bbParams[0].conv_0_weight = bb_0_conv_0_weight;
bbParams[0].conv_1_weight = bb_0_conv_1_weight;
// Basic Block 1
bbParams[1].conv_0_weight = bb_1_conv_0_weight;
bbParams[1].conv_1_weight = bb_1_conv_1_weight;
bbParams[1].conv_skip_weight = bb_1_conv_skip_weight;
// Basic Block 2
bbParams[2].conv_0_weight = bb_2_conv_0_weight;
bbParams[2].conv_1_weight = bb_2_conv_1_weight;
// Basic Block 3
bbParams[3].conv_0_weight = bb_3_conv_0_weight;
bbParams[3].conv_1_weight = bb_3_conv_1_weight;
bbParams[3].conv_skip_weight = bb_3_conv_skip_weight;
// Basic Block 4
bbParams[4].conv_1_weight = bb_4_conv_1_weight;


#endif


