/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-04 11:15:06
 * @FilePath: /txy/Garnet/GPU/test.cpp
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
#include "gpu.h"
#include "Math/bigint.h"
#include "Protocols/Fss3Prep.h"

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTES 8
int main(){
    
    bigint tmp[2], s[2][2], v[2][2],  t[2][2];
    octet seed[2][2*(2*LAMBDA_BYTES+1)], result[LAMBDA_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES];
    uint8_t res[LAMBDA_BYTES];
    // generate key
    tmp[0] = 6213237549190506053;
    tmp[1] = 8248733145573590288;
    bytesFromBigint(&a[0], tmp[0], LAMBDA_BYTES);
    bytesFromBigint(&b[0], tmp[1], LAMBDA_BYTES);
    test_add(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[0] + tmp[1] << std::endl;

    test_restricted_multiply(1, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[1] << std::endl;

    test_xor(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    
    PRNG prng;
    bigint r0, r1, r;
    int bit_length = LAMBDA_BYTES * 8;
    octet generate_value[2][(bit_length - 1) * 2 * (LAMBDA_BYTES + 1) + LAMBDA_BYTES];
    prng.InitSeed();
    prng.get(r0, LAMBDA_BYTES);
    prng.get(r1, LAMBDA_BYTES);
    octet r_inp[LAMBDA_BYTES], r_share[2][LAMBDA_BYTES];
    bytesFromBigint(&r_share[0][0], r0, LAMBDA_BYTES);
    bytesFromBigint(&r_share[1][0], r1, LAMBDA_BYTES);
    bytesFromBigint(&r_inp[0], r, LAMBDA_BYTES);
    //uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * key, uint8_t * generated_value_cpu_0, uint8_t * generated_value_cpu_1, int numbytes
    uint8_t key[32];
    int len = sizeof(key);
    for (int i = 0; i < len; i++){
        key[i] = i;
    }

    printf("CPU preparing finished!");
    fss_generate(r_inp, r_share[0], r_share[1], key, generate_value[0], generate_value[1], LAMBDA_BYTES);
    return 0;
}