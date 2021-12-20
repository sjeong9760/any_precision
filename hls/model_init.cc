#include <stdlib.h>

#include "model_init.h"
#include "model_param.h"
#include "model_shape.h"

void init_model(
  void* _bb_0_bn,
  void* _bb_0_conv_bn_weight,
  void* _bb_0_conv_bn_alpha_beta,
  void* _bb_0_conv_weight,

  void* _bb_1_bn,
  void* _bb_1_conv_bn_weight,
  void* _bb_1_conv_bn_alpha_beta,
  void* _bb_1_conv_weight,
  void* _bb_1_skip_conv_bn_weight,
  void* _bb_1_skip_conv_bn_alpha_beta,

  void* _bb_2_bn,
  void* _bb_2_conv_bn_weight,
  void* _bb_2_conv_bn_alpha_beta,
  void* _bb_2_conv_weight,

  void* _bb_3_bn,
  void* _bb_3_conv_bn_weight,
  void* _bb_3_conv_bn_alpha_beta,
  void* _bb_3_conv_weight,
  void* _bb_3_skip_conv_bn_weight,
  void* _bb_3_skip_conv_bn_alpha_beta,

  void* _bb_7_bn,
  void* _bb_7_conv_weight,

  void* _outer_bn,
  void* _outer_fc_weight,
  void* _outer_fc_bias) {

  memcpy((void*)bb_0_conv_0_weight, _bb_0_conv_0_weight, sizeof(WEIGHT_T) * BB_0_CONV_0_WEIGHT_LEN);
  memcpy((void*)bb_0_conv_1_weight, _bb_0_conv_1_weight, sizeof(WEIGHT_T) * BB_0_CONV_1_WEIGHT_LEN);

  memcpy((void*)bb_1_conv_0_weight, _bb_1_conv_0_weight, sizeof(WEIGHT_T) * BB_1_CONV_0_WEIGHT_LEN);
  memcpy((void*)bb_1_conv_1_weight, _bb_1_conv_1_weight, sizeof(WEIGHT_T) * BB_1_CONV_1_WEIGHT_LEN);
  memcpy((void*)bb_1_conv_skip_weight, _bb_1_conv_skip_weight, sizeof(WEIGHT_T) * BB_1_CONV_SKIP_WEIGHT_LEN);

  memcpy((void*)bb_2_conv_0_weight, _bb_2_conv_0_weight, sizeof(WEIGHT_T) * BB_2_CONV_0_WEIGHT_LEN);
  memcpy((void*)bb_2_conv_1_weight, _bb_2_conv_1_weight, sizeof(WEIGHT_T) * BB_2_CONV_1_WEIGHT_LEN);
  
  memcpy((void*)bb_3_conv_0_weight, _bb_3_conv_0_weight, sizeof(WEIGHT_T) * BB_3_CONV_0_WEIGHT_LEN);
  memcpy((void*)bb_3_conv_1_weight, _bb_3_conv_1_weight, sizeof(WEIGHT_T) * BB_3_CONV_1_WEIGHT_LEN);
  memcpy((void*)bb_3_conv_skip_weight, _bb_3_conv_skip_weight, sizeof(WEIGHT_T) * BB_3_CONV_SKIP_WEIGHT_LEN);

  memcpy((void*)bb_4_conv_1_weight, _bb_4_conv_1_weight, sizeof(WEIGHT_T) * BB_4_CONV_1_WEIGHT_LEN);

  memcpy((void*)fc_weight, _fc_weight, sizeof(WEIGHT_T) * FC_WEIGHT_LEN);
  memcpy((void*)fc_bias, _fc_bias, sizeof(float) * otherShapes.fc_bias_shape);
}
