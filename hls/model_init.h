#ifndef _MODEL_INIT_H
#define _MODEL_INIT_H

void init_model(
  void* _bb_0_conv_0_weight,
  void* _bb_0_conv_1_weight,

  void* _bb_1_conv_0_weight,
  void* _bb_1_conv_1_weight,
  void* _bb_1_conv_skip_weight,

  void* _bb_2_conv_0_weight,
  void* _bb_2_conv_1_weight,

  void* _bb_3_conv_0_weight,
  void* _bb_3_conv_1_weight,
  void* _bb_3_conv_skip_weight,

  void* _bb_4_conv_1_weight,

  void* _fc_weight,
  void* _fc_bias);

#endif
