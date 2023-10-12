/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-12 18:32:44
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
    int bit_length = LAMBDA_BYTES * 8;
    bigint tmp, s[2][2], v[2][2],  t[2][2], _t, scw, vcw, tcw[2];
    octet result[LAMBDA_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES], c[1], offset_reveal[LAMBDA_BYTES], result1[LAMBDA_BYTES], result2[LAMBDA_BYTES], result3[LAMBDA_BYTES], result4[LAMBDA_BYTES];

    PRNG prng;
    bigint seed0, seed1, r, x, x_add_r, eval_result[4], final_res;
    octet generate_value[(bit_length) * 2 * (LAMBDA_BYTES + 1) + LAMBDA_BYTES];
    prng.InitSeed();
    x_add_r = 1203812904;
    // prng.get(x_add_r, bit_length);
    x = 0;
    r = x_add_r - x;
    seed0 = 124812904;
    seed1 = 193298434;
    // prng.get(seed0, bit_length);
    // prng.get(seed1, bit_length);
    octet r_inp[LAMBDA_BYTES], x_reveal[LAMBDA_BYTES], seed[2][LAMBDA_BYTES];
    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bytesFromBigint(&seed[1][0], seed1, LAMBDA_BYTES);
    bytesFromBigint(&r_inp[0], r, LAMBDA_BYTES);
    bytesFromBigint(&x_reveal[0], x_add_r, LAMBDA_BYTES);
            
    if(x_reveal[0] >= 128){
        std::cout << "x_reveal 0 >= 128" << std::endl;
        final_res = 1;
        offset_reveal[0] = x_reveal[0] - 128;
    }
    else{
        final_res = 0;
        offset_reveal[0] = x_reveal[0] + 128;
    }
    for(int i = 1; i < LAMBDA_BYTES; i++){
        offset_reveal[i] = x_reveal[i];
    }


    printf("CPU preparing finished!\n");

    //uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel
    fss_generate(r_inp, seed[0], seed[1], generate_value, LAMBDA_BYTES, 1);
    // for(int i = 0; i < bit_length - 1; i++){
    //     bigintFromBytes(scw, &generate_value[2*(LAMBDA_BYTES+1)*i], LAMBDA_BYTES);
    //     bigintFromBytes(vcw, &generate_value[2*(LAMBDA_BYTES+1)*i + LAMBDA_BYTES], LAMBDA_BYTES);
    //     bigintFromBytes(tcw[0], &generate_value[2*(LAMBDA_BYTES+1)*i + 2 * LAMBDA_BYTES] , 1);
    //     bigintFromBytes(tcw[1], &generate_value[2*(LAMBDA_BYTES+1)*i + 2 * LAMBDA_BYTES + 1], 1);
    //     std::cout << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << std::endl;
    // }
    
    bigintFromBytes(vcw, &generate_value[(bit_length) * 2 * (LAMBDA_BYTES + 1)], LAMBDA_BYTES);
    std::cout << "final cw is " << vcw << std::endl;

    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bigintFromBytes(seed0, &seed[0][0], LAMBDA_BYTES);
    std::cout << "seed is " << seed0 << std::endl;
    fss_evaluate(0, x_reveal, seed[0], generate_value, result1, LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[0], &result1[0], LAMBDA_BYTES);
    final_res = final_res - eval_result[0];
    std::cout << eval_result[0] << std::endl;
    std::cout << "---------------" << std::endl;

    bytesFromBigint(&seed[0][0], seed0, LAMBDA_BYTES);
    bigintFromBytes(seed0, &seed[0][0], LAMBDA_BYTES);
    std::cout << "seed is " << seed0 << std::endl;
    fss_evaluate(0, offset_reveal, seed[0], generate_value, result2, LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[1], &result2[0], LAMBDA_BYTES);
    std::cout << eval_result[1] << std::endl;
    final_res = final_res + eval_result[1];
    std::cout << "---------------" << std::endl;

    fss_evaluate(1, x_reveal, seed[1], generate_value, result3, LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[2], &result3[0], LAMBDA_BYTES);
    std::cout << eval_result[2] << std::endl;
    final_res = final_res - eval_result[2];
    std::cout << "---------------" << std::endl;

    fss_evaluate(1, offset_reveal, seed[1], generate_value, result4, LAMBDA_BYTES, 1);
    bigintFromBytes(eval_result[3], &result4[0], LAMBDA_BYTES);
    std::cout << eval_result[3] << std::endl;
    final_res = final_res + eval_result[3];
    std::cout << final_res << std::endl;
    std::cout << "---------------" << std::endl;
    return 0;
}