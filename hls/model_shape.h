#ifndef _MODEL_SHAPE_H
#define _MODEL_SHAPE_H

#include <stdint.h>

// Parameter shapes of basic blocks
typedef struct BasicBlockShapes {
  uint16_t bn_alpha_shape;

  uint16_t skip_conv_bn_alpha_shape;
  uint16_t skip_conv_bn_weight_shape[4]; // NHWC
  uint16_t skip_conv_bn_stride[2];
  uint16_t skip_conv_bn_padding[2];

  uint16_t conv_bn_alpha_shape;
  uint16_t conv_bn_weight_shape[4];      // NHWC
  uint16_t conv_bn_stride[2];
  uint16_t conv_bn_padding[2];

  uint16_t conv_weight_shape[4];         // NHWC
  uint16_t conv_stride[2];
  uint16_t conv_padding[2];
} BasicBlockShapes;

// Parameter shapes of other blocks (e.g., out of basic blocks)
typedef struct OtherShapes {
  uint16_t bn_alpha_shape;
  uint16_t fc_weight_shape[2];
  uint16_t fc_bias_shape;
} OtherShapes;

const uint16_t NUM_BASIC_BLOCKS = 5;
const uint16_t NUM_CLASSES = 10;
const BasicBlockShapes bbShapes[NUM_BASIC_BLOCKS] = {
  // Layer 0
  { 80,                   // skip_conv_bn_alpha_beta 
    0,  {0, 0, 0, 0},      // skip_conv_bn_alph_beta, skip_conv_bn_weight 
    {0, 0}, {0, 0},       // skip_conv_bn_stride, skip_conv_bn_padding
    80, {80, 3, 3, 80}, // conv_bn_alpha_beta, conv_bn_weight
    {1, 1}, {1, 1},       // conv_bn_stride, conv_bn_padding
    {80, 3, 3, 80},       // conv_weight
    {1, 1}, {1, 1}        // conv_stride, conv_padding
  },
  // Layer 1
  { 80,   
    160, {160, 1, 1, 80}, 
    {2, 2}, {0, 0},
    160, {160, 3, 3, 80}, 
    {2, 2}, {1, 1},
    {160, 3, 3, 160}, 
    {1, 1}, {1, 1}
  },
  // Layer 2
  { 160,  
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
    160,  {160, 3, 3, 160}, 
    {1, 1}, {1, 1},
    {160, 3, 3, 160}, 
    {1, 1}, {1, 1}
  },
  // Layer 3
  { 160,  
    320,  {320, 1, 1, 160}, 
    {2, 2}, {0, 0},
    320,  {320, 3, 3, 160}, 
    {2, 2}, {1, 1},
    {320, 3, 3, 320}, 
    {1, 1}, {1, 1}
  },
  // Layer 4
  { 320,  
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
    {320, 3, 3, 320}, 
    {1, 1}, {1, 1}
  }
};
const OtherShapes otherShapes = {
//  bn    fc_weight            fc_bias
    320,  {NUM_CLASSES, 320},  NUM_CLASSES
};

#endif
