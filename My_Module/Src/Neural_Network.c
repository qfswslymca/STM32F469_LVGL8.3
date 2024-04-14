/*
 * Neural_Network.c
 *
 *  Created on: Apr 12, 2024
 *      Author: 21536
 */

#include "Neural_Network.h"
#include "arm_nn_types.h"
#include "arm_nn_compiler.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "LeNet5_quantized_params_conv1.h"
#include "LeNet5_quantized_params_conv2.h"
#include "LeNet5_quantized_params_conv3.h"
#include "LeNet5_quantized_params_fc1.h"
#include "LeNet5_quantized_params_fc2.h"
#include "LeNet5_quantized_params_data.h"
#include "LeNet5_quantized_params_pool1.h"
#include "LeNet5_quantized_params_pool2.h"

#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdlib.h>
#include <stdio.h>
#include <lvgl.h>
#include "./demos/lv_demos.h"
#include <screen_driver.h>
#include <touch_sensor_driver.h>

extern int8_t grayscale[28*28];

int findMaxIndex(int8_t array[], int length) {
    if (length <= 0)
        return -1; // 返回-1表示数组为空或长度非法

    int maxIndex = 0; // 假设最大值索引为第一个元素的索引
    for (int i = 1; i < length; ++i) {
        if (array[i] > array[maxIndex])
            maxIndex = i; // 找到更大的值，更新最大值索引
    }

    return maxIndex; // 返回最大值索引加1
}

void LeNet5(int8_t grayscale[28*28], int8_t* pred_num){

//    int8_t conv1_input[] = grayscale;
    int8_t conv1_kernel[] = CONV1_KERNEL;
    int32_t conv1_bias[] = CONV1_BIAS;
    int8_t conv2_kernel[] = CONV2_KERNEL;
    int32_t conv2_bias[] = CONV2_BIAS;
    int8_t conv3_kernel[] = CONV3_KERNEL;
    int32_t conv3_bias[] = CONV3_BIAS;
    int8_t fc1_kernel[] = FC1_WEIGHT;
    int32_t fc1_bias[] = FC1_BIAS;
    int8_t fc2_kernel[] = FC2_WEIGHT;
    int32_t fc2_bias[] = FC2_BIAS;
    int32_t conv1_quant_param_multi[] = CONV1_QUANT_PARAM_MULTI;
    int32_t conv1_quant_param_shift[] = CONV1_QUANT_PARAM_SHIFT;
    int32_t conv2_quant_param_multi[] = CONV2_QUANT_PARAM_MULTI;
    int32_t conv2_quant_param_shift[] = CONV2_QUANT_PARAM_SHIFT;
    int32_t conv3_quant_param_multi[] = CONV3_QUANT_PARAM_MULTI;
    int32_t conv3_quant_param_shift[] = CONV3_QUANT_PARAM_SHIFT;

    const cmsis_nn_conv_params conv1_param = CONV1_PARAM;
    const cmsis_nn_per_channel_quant_params conv1_quant_param = {conv1_quant_param_multi, conv1_quant_param_shift}; // {multiplier, shift}
    const cmsis_nn_conv_params conv2_param = CONV2_PARAM;
    const cmsis_nn_per_channel_quant_params conv2_quant_param = {conv2_quant_param_multi, conv2_quant_param_shift}; // {multiplier, shift}
    const cmsis_nn_conv_params conv3_param = CONV3_PARAM;
    const cmsis_nn_per_channel_quant_params conv3_quant_param = {conv3_quant_param_multi, conv3_quant_param_shift}; // {multiplier, shift}
    const cmsis_nn_pool_params pool1_param = POOL1_PARAM;
    const cmsis_nn_pool_params pool2_param = POOL2_PARAM;
    const cmsis_nn_fc_params fc1_param = FC1_PARAM;
    const cmsis_nn_per_tensor_quant_params fc1_quant_param = FC1_QUANT_PARAM;
    const cmsis_nn_fc_params fc2_param = FC2_PARAM;
    const cmsis_nn_per_tensor_quant_params fc2_quant_param = FC2_QUANT_PARAM;


    const cmsis_nn_dims conv1_input_dims = {1, 28, 28, 1}; // n h w c
    const cmsis_nn_dims conv1_filter_dims = {6, 5, 5, 1};
    const cmsis_nn_dims conv1_bias_dims = {6, 1, 1, 1};
    const cmsis_nn_dims conv1_output_dims = {1, 28, 28, 6};
    const cmsis_nn_dims poo1_filter_dims = {1, 2, 2, 6};

    const cmsis_nn_dims conv2_input_dims = {1, 14, 14, 6};
    const cmsis_nn_dims conv2_filter_dims = {16, 5, 5, 6};
    const cmsis_nn_dims conv2_bias_dims = {16, 1, 1, 1};
    const cmsis_nn_dims conv2_output_dims = {1, 10, 10, 16};
    const cmsis_nn_dims poo2_filter_dims = {1, 2, 2, 16};

    const cmsis_nn_dims conv3_input_dims = {1, 5, 5, 16};
    const cmsis_nn_dims conv3_filter_dims = {120, 5, 5, 16};
    const cmsis_nn_dims conv3_bias_dims = {120, 1, 1, 1};
    const cmsis_nn_dims conv3_output_dims = {1, 1, 1, 120};

    const cmsis_nn_dims fc1_input_dims = {1, 1, 1, 120}; // n h w c
    const cmsis_nn_dims fc1_filter_dims = {120, 1, 1, 84};
    const cmsis_nn_dims fc1_bias_dims = {84, 1, 1, 1};

    const cmsis_nn_dims fc2_input_dims = {1, 1, 1, 84};
    const cmsis_nn_dims fc2_filter_dims = {84, 1, 1, 10};
    const cmsis_nn_dims fc2_bias_dims = {10, 1, 1, 1};
    const cmsis_nn_dims fc2_output_dims = {1, 1, 1, 10};

    int8_t conv1_output[6*28*28];
    int8_t conv2_input[14*14*6];
    int8_t conv2_output[10*10*16];
    int8_t conv3_input[5*5*16];
    int8_t conv3_output[120];

    int8_t fc1_input[120];
    int8_t fc2_input[84];
    int8_t fc2_output[10];
    int8_t result[10];

    int8_t buf[1000];
    cmsis_nn_context ctx = {buf, 1000};

    arm_convolve_wrapper_s8(&ctx,
                    &conv1_param,
                    &conv1_quant_param,
                    &conv1_input_dims,
                    grayscale,
                    &conv1_filter_dims,
                    conv1_kernel,
                    &conv1_bias_dims,
                    conv1_bias,
                    &conv1_output_dims,
                    conv1_output);


    arm_avgpool_s8(&ctx,
                    &pool1_param,
                    &conv1_output_dims,
                    conv1_output,
                    &poo1_filter_dims,
                    &conv2_input_dims,
                    conv2_input);

    arm_convolve_wrapper_s8(&ctx,
                    &conv2_param,
                    &conv2_quant_param,
                    &conv2_input_dims,
                    conv2_input,
                    &conv2_filter_dims,
                    conv2_kernel,
                    &conv2_bias_dims,
                    conv2_bias,
                    &conv2_output_dims,
                    conv2_output);

    arm_avgpool_s8(&ctx,
                    &pool2_param,
                    &conv2_output_dims,
                    conv2_output,
                    &poo2_filter_dims,
                    &conv3_input_dims,
                    conv3_input);

    arm_convolve_wrapper_s8(&ctx,
                    &conv3_param,
                    &conv3_quant_param,
                    &conv3_input_dims,
                    conv3_input,
                    &conv3_filter_dims,
                    conv3_kernel,
                    &conv3_bias_dims,
                    conv3_bias,
                    &conv3_output_dims,
                    conv3_output);

    arm_fully_connected_s8(&ctx,
                        &fc1_param,
                        &fc1_quant_param,
                        &fc1_input_dims,
                        conv3_output,
                        &fc1_filter_dims,
                        fc1_kernel,
                        &fc1_bias_dims,
                        fc1_bias,
                        &fc2_input_dims,
                        fc2_input
                        );

    arm_fully_connected_s8(&ctx,
                        &fc2_param,
                        &fc2_quant_param,
                        &fc2_input_dims,
                        fc2_input,
                        &fc2_filter_dims,
                        fc2_kernel,
                        &fc2_bias_dims,
                        fc2_bias,
                        &fc2_output_dims,
                        fc2_output);

    arm_nn_softmax_common_s8(fc2_output,
                   fc2_output_dims.n,
                   fc2_output_dims.c,
                   fc2_quant_param.multiplier,
                   fc2_quant_param.shift + 31,
                   -256, false,
                   result);

    *pred_num = findMaxIndex(result, 10);
}
