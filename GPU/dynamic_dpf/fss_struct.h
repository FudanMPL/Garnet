/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-24 10:34:10
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-11-04 10:40:24
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
#include <vector>
// #include <cuda.h>

#ifndef FSS_STRUCT_H_
#define FSS_STRUCT_H_

#define INPUT_BYTE 8
#define LAMBDA_BYTE 16

class aes_gen_block
{
public:
    uint8_t block[2][2*LAMBDA_BYTE];
};

class aes_eval_block
{
public:
    uint8_t block[2*LAMBDA_BYTE];
};

class KeyBlock{
public:
    uint8_t cuda_key[240];
};

class FssDpfGen
{
public:
    uint8_t s[2][2][LAMBDA_BYTE];
    bool t[2][2];
    bool pre_t[2]; 
    bool keep;
    bool lose;
};

class FssDpfEval
{
public:
    uint8_t s[2][LAMBDA_BYTE];
    bool t[2];
    bool pre_t; 
    bool xi;
};

struct InputByteRelatedValuesGen{
    uint8_t * r;
    uint8_t * r_share_0;
    uint8_t * r_share_1;
    uint8_t * scw;
    bool * tcw[2];
    uint8_t * output;
};

struct InputByteRelatedValuesEval{
    uint8_t * r_share;
    uint8_t * scw;
    bool * tcw[2];
    uint8_t * output;
    uint8_t * result;
};


#endif