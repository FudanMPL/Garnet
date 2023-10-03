/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-03 11:32:45
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-03 12:05:09
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

struct fss_gen_struct
{
    uint8_t * seed[2];
    uint8_t * r;
    uint8_t * s[2];
    uint8_t * v[2];
    uint8_t * t[2];
    uint8_t * pre_t[2];
    uint8_t * scw[2];
    uint8_t * vcw[2];
    uint8_t * tmp[2];
    uint8_t * convert[2];
    uint8_t * va;
    int * keep, * lose;
};

// struct fss_eval_struct
// {

// }
