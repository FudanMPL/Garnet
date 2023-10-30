/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-11 09:06:57
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-11 09:07:10
 * @FilePath: /txy/Garnet/GPU/test_utils.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include "gpu.h"
#include "Math/bigint.h"
#include "Protocols/Fss3Prep.h"

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTES 8
int main(){
    
    bigint tmp[2], s[2][2], v[2][2],  t[2][2], _t, scw, vcw, tcw[2];
    octet result[LAMBDA_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES], c[1];
    uint8_t res[LAMBDA_BYTES];
    tmp[0] = 6213237549190506053;
    tmp[1] = 8248733145573590288;
    bytesFromBigint(&a[0], tmp[0], LAMBDA_BYTES);
    bytesFromBigint(&b[0], tmp[1], LAMBDA_BYTES);
    test_add(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[0] + tmp[1] << std::endl;

    test_sub(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[0] - tmp[1] << std::endl;

    tmp[1] = 8248733145573590288;
    bytesFromBigint(&b[0], tmp[1], LAMBDA_BYTES);
    
    test_restricted_multiply(1, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << "test_multiply" << std::endl;
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[1] << std::endl;

    test_xor(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
}