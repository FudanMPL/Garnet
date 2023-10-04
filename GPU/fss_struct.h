/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-03 11:32:45
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-04 15:01:23
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

struct fss_gen_struct_64
{
    uint8_t seed[2][8];
    uint8_t r[8];
    uint8_t s[2][2][8];
    uint8_t v[2][2][8];
    uint8_t t[2][2][8];
    uint8_t pre_t[2][2][8];
    uint8_t scw[2][8];
    uint8_t vcw[2][8];
    uint8_t tcw[2][8];
    uint8_t inter_val[2][8];
    uint8_t convert[2][8];
    uint8_t convert_seed[2][8];
    uint8_t va[8];
    int keep, lose;
};

struct fss_gen_struct_128
{
    uint8_t seed[2][16];
    uint8_t r[16];
    uint8_t s[2][2][16];
    uint8_t v[2][2][16];
    uint8_t t[2][2][16];
    uint8_t pre_t[2][2][16];
    uint8_t scw[2][16];
    uint8_t vcw[2][16];
    uint8_t tcw[2][16];
    uint8_t inter_val[2][16];
    uint8_t convert[2][16];
    uint8_t convert_seed[2][16];
    uint8_t va[16];
    int keep, lose;
};

// struct fss_eval_struct
// {

// }

#endif