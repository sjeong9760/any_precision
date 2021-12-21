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
  void* _bb_1_skip_conv_bn_weight,
  void* _bb_1_skip_conv_bn_alpha_beta,
  void* _bb_1_conv_bn_weight,
  void* _bb_1_conv_bn_alpha_beta,
  void* _bb_1_conv_weight,

  void* _bb_2_bn,
  void* _bb_2_conv_bn_weight,
  void* _bb_2_conv_bn_alpha_beta,
  void* _bb_2_conv_weight,

  void* _bb_3_bn,
  void* _bb_3_skip_conv_bn_weight,
  void* _bb_3_skip_conv_bn_alpha_beta,
  void* _bb_3_conv_bn_weight,
  void* _bb_3_conv_bn_alpha_beta,
  void* _bb_3_conv_weight,

  void* _bb_4_bn,
  void* _bb_4_conv_weight,

  void* _outer_bn,
  void* _outer_fc_weight,
  void* _outer_fc_bias) {

  memcpy((void*)bb_0_bn_alpha_beta, _bb_0_bn, sizeof(float) * BB_0_ALPHA_BETA_LEN);
  memcpy((void*)bb_0_conv_bn_alpha_beta, _bb_0_conv_bn_alpha_beta, sizeof(float) * BB_0_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_0_conv_bn_weight, _bb_0_conv_bn_weight, sizeof(WEIGHT_T) * BB_0_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_0_conv_weight, _bb_0_conv_weight, sizeof(WEIGHT_T) * BB_0_CONV_WEIGHT_LEN);

  memcpy((void*)bb_1_bn_alpha_beta, _bb_1_bn, sizeof(float) * BB_1_ALPHA_BETA_LEN);
  memcpy((void*)bb_1_conv_bn_alpha_beta, _bb_1_conv_bn_alpha_beta, sizeof(float) * BB_1_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_1_conv_bn_weight, _bb_1_conv_bn_weight, sizeof(WEIGHT_T) * BB_1_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_1_skip_conv_bn_alpha_beta, _bb_1_skip_conv_bn_alpha_beta, sizeof(float) * BB_1_SKIP_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_1_skip_conv_bn_weight, _bb_1_skip_conv_bn_weight, sizeof(WEIGHT_T) * BB_1_SKIP_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_1_conv_weight, _bb_1_conv_weight, sizeof(WEIGHT_T) * BB_1_CONV_WEIGHT_LEN);

  memcpy((void*)bb_2_bn_alpha_beta, _bb_2_bn, sizeof(float) * BB_2_ALPHA_BETA_LEN);
  memcpy((void*)bb_2_conv_bn_alpha_beta, _bb_2_conv_bn_alpha_beta, sizeof(float) * BB_2_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_2_conv_bn_weight, _bb_2_conv_bn_weight, sizeof(WEIGHT_T) * BB_2_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_2_conv_weight, _bb_2_conv_weight, sizeof(WEIGHT_T) * BB_2_CONV_WEIGHT_LEN);

  memcpy((void*)bb_3_bn_alpha_beta, _bb_3_bn, sizeof(float) * BB_3_ALPHA_BETA_LEN);
  memcpy((void*)bb_3_conv_bn_alpha_beta, _bb_3_conv_bn_alpha_beta, sizeof(float) * BB_3_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_3_conv_bn_weight, _bb_3_conv_bn_weight, sizeof(WEIGHT_T) * BB_3_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_3_skip_conv_bn_alpha_beta, _bb_3_skip_conv_bn_alpha_beta, sizeof(float) * BB_3_SKIP_CONV_BN_ALPHA_BETA_LEN);
  memcpy((void*)bb_3_skip_conv_bn_weight, _bb_3_skip_conv_bn_weight, sizeof(WEIGHT_T) * BB_3_SKIP_CONV_BN_WEIGHT_LEN);
  memcpy((void*)bb_3_conv_weight, _bb_3_conv_weight, sizeof(WEIGHT_T) * BB_3_CONV_WEIGHT_LEN);

  memcpy((void*)bb_4_bn_alpha_beta, _bb_4_bn, sizeof(float) * BB_4_ALPHA_BETA_LEN);
  memcpy((void*)bb_4_conv_weight, _bb_4_conv_weight, sizeof(WEIGHT_T) * BB_4_CONV_WEIGHT_LEN);

  memcpy((void*)outer_bn, _outer_bn, sizeof(float) * OUTER_ALPHA_BETA_LEN);
  memcpy((void*)outer_fc_weight, _outer_fc_weight, sizeof(WEIGHT_T) * OUTER_FC_WEIGHT_LEN);
  memcpy((void*)outer_fc_bias, _outer_fc_bias, sizeof(float) * OUTER_FC_BIAS_LEN);
}
