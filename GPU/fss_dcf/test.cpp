/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-23 16:49:35
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
#define LAMBDA_BYTES 16
#define N_BYTES 8


int main(){
    int lambda = 127;
    int bit_length = N_BYTES * 8;

    bigint tmp, s[2][2], v[2][2],  t[2][2], _t, scw, vcw, tcw[2];
    octet result[N_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES], c[1], offset_reveal[N_BYTES], result1[N_BYTES], result2[N_BYTES], result3[N_BYTES], result4[N_BYTES];

    PRNG prng;
    bigint seed0, seed1, r, x, x_add_r, eval_result[4], final_res;
    octet generate_value[((bit_length) * 2 * LAMBDA_BYTES + N_BYTES) * sizeof(uint8_t)];
    prng.InitSeed();
    prng.get(x_add_r, bit_length);
    prng.get(x, bit_length);
    r = x_add_r - x;
    seed0 = 124812904;
    seed1 = 193298434;
    // prng.get(seed0, bit_length);
    // prng.get(seed1, bit_length);
    octet r_inp[LAMBDA_BYTES], x_reveal[N_BYTES], seed[2][LAMBDA_BYTES];
    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bytesFromBigint(&seed[1][0], seed1, LAMBDA_BYTES);
    bytesFromBigint(&r_inp[0], r, N_BYTES);
    bytesFromBigint(&x_reveal[0], x_add_r, N_BYTES);
            
    if(x_reveal[0] >= 128){
        std::cout << "x_reveal 0 >= 128" << std::endl;
        final_res = 1;
        offset_reveal[0] = x_reveal[0] - 128;
    }
    else{
        final_res = 0;
        offset_reveal[0] = x_reveal[0] + 128;
    }
    for(int i = 1; i < N_BYTES; i++){
        offset_reveal[i] = x_reveal[i];
    }


    printf("CPU preparing finished!\n");
    std::cout << "x is " << x << std::endl;
    //uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel
    fss_generate(r_inp, seed[0], seed[1], generate_value, N_BYTES, 1);
    
    bigintFromBytes(vcw, &generate_value[(bit_length) * 2 * int((lambda + 1)/8)], LAMBDA_BYTES);
    std::cout << "final cw is " << vcw << std::endl;
    
    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bigintFromBytes(seed0, &seed[0][0], LAMBDA_BYTES);
    fss_evaluate(0, x_reveal, seed[0], generate_value, result1, N_BYTES, 1);
    bigintFromBytes(eval_result[0], &result1[0], N_BYTES);
    final_res = final_res - eval_result[0];
    
    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bigintFromBytes(seed0, &seed[0][0], LAMBDA_BYTES);
    fss_evaluate(0, offset_reveal, seed[0], generate_value, result2, N_BYTES, 1);
    bigintFromBytes(eval_result[1], &result2[0], N_BYTES);
    final_res = final_res + eval_result[1];

    bytesFromBigint(&seed[1][0], seed1, LAMBDA_BYTES);
    bigintFromBytes(seed1, &seed[1][0], LAMBDA_BYTES);
    fss_evaluate(1, x_reveal, seed[1], generate_value, result3, N_BYTES, 1);
    bigintFromBytes(eval_result[2], &result3[0], N_BYTES);
    final_res = final_res - eval_result[2];
    
    
    fss_evaluate(1, offset_reveal, seed[1], generate_value, result4, N_BYTES, 1);
    bigintFromBytes(eval_result[3], &result4[0], LAMBDA_BYTES);
    final_res = final_res + eval_result[3];

    if(offset_reveal[0] >> 7)
        final_res = final_res + 1;
    std::cout << final_res << std::endl;
    return 0;
}