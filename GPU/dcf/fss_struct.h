/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-03 11:32:45
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-10 19:55:14
 * @FilePath: /txy/Garnet/GPU/fss_struct.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
#include <cuda.h>

#ifndef FSS_STRUCT_H_
#define FSS_STRUCT_H_

#define BYTE_LEN 8

class aes_block
{
public:
    uint8_t block[BYTE_LEN];
};


class FssGen
{
public:
    uint8_t seed[2][BYTE_LEN];
    uint8_t s[2][2][BYTE_LEN];
    uint8_t v[2][2][BYTE_LEN];
    uint8_t t[2][2][1];
    int pre_t[2];
    uint8_t scw[BYTE_LEN];
    uint8_t vcw[BYTE_LEN];
    uint8_t tcw[2][1];
    uint8_t inter_val[2][BYTE_LEN];
    uint8_t convert[2][BYTE_LEN];
    uint8_t convert_seed[2][BYTE_LEN];
    uint8_t va[BYTE_LEN];
    int keep, lose;
    
};

class FssEval
{
public:
    uint8_t seed[BYTE_LEN];
    uint8_t t_hat[2][1];
    uint8_t v_hat[2][BYTE_LEN];
    uint8_t s_hat[2][BYTE_LEN];
    uint8_t t[2][1];
    uint8_t s[2][BYTE_LEN];
    uint8_t tcw[2][1];
    uint8_t scw[BYTE_LEN];
    uint8_t vcw[BYTE_LEN];
    uint8_t inter_val[BYTE_LEN];
    uint8_t tmp_v[BYTE_LEN];
    int pre_t;
    uint8_t convert[2][BYTE_LEN];
    uint8_t convert_seed[BYTE_LEN];
};

#endif