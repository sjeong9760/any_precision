#ifndef _MODEL_INIT_H
#define _MODEL_INIT_H

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
  void* _outer_fc_bias);

#endif
