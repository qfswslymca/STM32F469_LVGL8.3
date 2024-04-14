/*
 * Neural_Network.h
 *
 *  Created on: Apr 12, 2024
 *      Author: 21536
 */

#ifndef INC_NEURAL_NETWORK_H_
#define INC_NEURAL_NETWORK_H_
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdlib.h>
#include <stdio.h>
#include <lvgl.h>
#include "./demos/lv_demos.h"
#include <screen_driver.h>
#include <touch_sensor_driver.h>

void LeNet5(int8_t grayscale[28*28], int8_t* pred_num);

#endif /* INC_NEURAL_NETWORK_H_ */
