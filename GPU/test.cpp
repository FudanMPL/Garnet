/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-10 23:06:31
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
    
    bigint tmp[2], s[2][2], v[2][2],  t[2][2], _t, scw, vcw, tcw[2];
    octet result[LAMBDA_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES], c[1];
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
    
    PRNG prng;
    bigint seed0, seed1, r, x, x_add_r, eval_result[2];
    int bit_length = LAMBDA_BYTES * 8;
    octet generate_value[(bit_length - 1) * 2 * (LAMBDA_BYTES + 1) + LAMBDA_BYTES];
    prng.InitSeed();
    prng.get(x_add_r, bit_length);
    x = 1;
    r = x_add_r - x;
    prng.get(seed0, bit_length);
    prng.get(seed1, bit_length);
    octet r_inp[LAMBDA_BYTES], x_reveal[LAMBDA_BYTES], seed[2][LAMBDA_BYTES];
    octet result_bytes[2][LAMBDA_BYTES];
    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bytesFromBigint(&seed[1][0], seed1, LAMBDA_BYTES);
    bytesFromBigint(&r_inp[0], r, LAMBDA_BYTES);
    bytesFromBigint(&x_reveal[0], x_add_r, LAMBDA_BYTES);
    

    printf("CPU preparing finished!\n");

    //uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel
    fss_generate(r_inp, seed[0], seed[1], generate_value, LAMBDA_BYTES, 1);
    for(int i = 0; i < bit_length - 1; i++){
        bigintFromBytes(scw, &generate_value[2*(LAMBDA_BYTES+1)*i], LAMBDA_BYTES);
        bigintFromBytes(vcw, &generate_value[2*(LAMBDA_BYTES+1)*i + LAMBDA_BYTES], LAMBDA_BYTES);
        bigintFromBytes(tcw[0], &generate_value[2*(LAMBDA_BYTES+1)*i + 2 * LAMBDA_BYTES] , 1);
        bigintFromBytes(tcw[1], &generate_value[2*(LAMBDA_BYTES+1)*i + 2 * LAMBDA_BYTES + 1], 1);
        std::cout << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << std::endl;
    }
    
    fss_evaluate(0, x_reveal, seed[0], generate_value, result_bytes[0], LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[0], &result_bytes[0][0], LAMBDA_BYTES);
    std::cout << eval_result[0] << std::endl;
    fss_evaluate(1, x_reveal, seed[1], generate_value, result_bytes[0], LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[1], &result_bytes[1][0], LAMBDA_BYTES);
    std::cout << eval_result[1] << std::endl;
    return 0;
}