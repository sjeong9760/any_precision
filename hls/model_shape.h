#ifndef _MODEL_SHAPE_H
#define _MODEL_SHAPE_H

#include <stdint.h>

// Parameter shapes of basic blocks
typedef struct BasicBlockShapes {
  uint16_t bn_0_shape;
  uint16_t bn_1_shape;
  uint16_t bn_skip_shape;
  uint16_t conv_0_shape[4];     // NHWC
  uint16_t conv_1_shape[4];     // NHWC
  uint16_t conv_skip_shape[4];  // NHWC
} BasicBlockShapes;

// Parameter shapes of other blocks (e.g., out of basic blocks)
typedef struct OtherShapes {
  uint16_t bn_shape;
  uint16_t fc_weight_shape[2];
  uint16_t fc_bias_shape;
} OtherShapes;

const uint16_t NUM_BASIC_BLOCKS = 5;
const uint16_t NUM_CLASSES = 10;
const BasicBlockShapes bbShapes[NUM_BASIC_BLOCKS] = {
//  bn_0  bn_1  bn_skip   conv_0            conv_1            conv_skip
  { 80,   80,   0,        {80, 3, 3, 80},   {80, 3, 3, 80},   {0, 0, 0, 0}      },  // Basic Block 0
  { 80,   160,  160,      {160, 3, 3, 80},  {160, 3, 3, 160}, {160, 1, 1, 80}   },  // Basic Block 1
  { 160,  160,  0,        {160, 3, 3, 160}, {160, 3, 3, 160}, {0, 0, 0, 0}      },  // Basic Block 2
  { 160,  320,  320,      {320, 3, 3, 160}, {320, 3, 3, 320}, {320, 1, 1, 160}  },  // Basic Block 3
  { 320,  0,    0,        {0, 0, 0, 0},     {320, 3, 3, 320}, {0, 0, 0, 0}      }   // Basic Block 4
};
const OtherShapes otherShapes = {
//  bn    fc_weight            fc_bias
    320,  {NUM_CLASSES, 320},  NUM_CLASSES
};

#endif
