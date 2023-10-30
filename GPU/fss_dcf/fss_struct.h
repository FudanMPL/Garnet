/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-03 11:32:45
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-23 14:03:04
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
#define BYTE_LAMBDA 16

class aes_block
{
public:
    uint8_t block[BYTE_LEN];
};


class FssGen
{
public:
    uint8_t seed[2][BYTE_LAMBDA];
    uint8_t s[2][2][BYTE_LAMBDA];
    uint8_t v[2][2][BYTE_LAMBDA];
    bool t[2][2];
    bool pre_t[2];
    uint8_t scw[BYTE_LAMBDA];
    uint8_t vcw[BYTE_LAMBDA];
    bool tcw[2];
    uint8_t inter_val[2][2*BYTE_LAMBDA];
    uint8_t convert[2][BYTE_LEN];
    uint8_t va[BYTE_LEN];    
};

class FssEval
{
public:
    uint8_t seed[BYTE_LAMBDA];
    bool t_hat[2];
    uint8_t v_hat[2][BYTE_LAMBDA];
    uint8_t s_hat[2][BYTE_LAMBDA];
    bool t[2];
    uint8_t s[2][BYTE_LAMBDA];
    bool tcw[2];
    uint8_t scw[BYTE_LAMBDA];
    uint8_t vcw[BYTE_LAMBDA];
    uint8_t inter_val[2][2*BYTE_LAMBDA];
    uint8_t tmp_v[BYTE_LEN];
    bool pre_t;
    uint8_t convert[2][BYTE_LEN];
};


#endif