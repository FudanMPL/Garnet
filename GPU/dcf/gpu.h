/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:06:03
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-10 22:45:52
 * @FilePath: /txy/Garnet/GPU/gpu.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "Math/bigint.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <vector>

void add(uint8_t *a, uint8_t *b, uint8_t *res, int length);

void restricted_multiply(int value, uint8_t * a, uint8_t *res, int length);

void test_add(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void test_sub(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void test_restricted_multiply(int value, uint8_t * a, uint8_t * res, int numbytes);

void test_xor(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel);

void fss_evaluate(int party, uint8_t * x_reveal, uint8_t * seed, uint8_t * gen_val, uint8_t * result, int numbytes, int parallel);