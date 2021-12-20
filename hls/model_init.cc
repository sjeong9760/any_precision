#include <stdlib.h>

#include "model_init.h"
#include "model_param.h"
#include "model_shape.h"

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
  void* _fc_bias) {

  memcpy((void*)bb_0_conv_0_weight, _bb_0_conv_0_weight, sizeof(ap_int<AP_SIZE>) * bb_0_conv_0_weight_len);
  memcpy((void*)bb_0_conv_1_weight, _bb_0_conv_1_weight, sizeof(ap_int<AP_SIZE>) * bb_0_conv_1_weight_len);

  memcpy((void*)bb_1_conv_0_weight, _bb_1_conv_0_weight, sizeof(ap_int<AP_SIZE>) * bb_1_conv_0_weight_len);
  memcpy((void*)bb_1_conv_1_weight, _bb_1_conv_1_weight, sizeof(ap_int<AP_SIZE>) * bb_1_conv_1_weight_len);
  memcpy((void*)bb_1_conv_skip_weight, _bb_1_conv_skip_weight, sizeof(ap_int<AP_SIZE>) * bb_1_conv_skip_weight_len);

  memcpy((void*)bb_2_conv_0_weight, _bb_2_conv_0_weight, sizeof(ap_int<AP_SIZE>) * bb_2_conv_0_weight_len);
  memcpy((void*)bb_2_conv_1_weight, _bb_2_conv_1_weight, sizeof(ap_int<AP_SIZE>) * bb_2_conv_1_weight_len);
  
  memcpy((void*)bb_3_conv_0_weight, _bb_3_conv_0_weight, sizeof(ap_int<AP_SIZE>) * bb_3_conv_0_weight_len);
  memcpy((void*)bb_3_conv_1_weight, _bb_3_conv_1_weight, sizeof(ap_int<AP_SIZE>) * bb_3_conv_1_weight_len);
  memcpy((void*)bb_3_conv_skip_weight, _bb_3_conv_skip_weight, sizeof(ap_int<AP_SIZE>) * bb_3_conv_skip_weight_len);

  memcpy((void*)bb_4_conv_1_weight, _bb_4_conv_1_weight, sizeof(ap_int<AP_SIZE>) * bb_4_conv_1_weight_len);

  memcpy((void*)fc_weight, _fc_weight, sizeof(ap_int<AP_SIZE>) * fc_weight_len);
  memcpy((void*)fc_bias, _fc_bias, sizeof(float) * otherShapes.fc_bias_shape);
}
