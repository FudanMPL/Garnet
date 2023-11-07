/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-24 16:24:02
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-11-07 10:06:58
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
#include "gpu.h"
#include <ctime>
#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTE 16

void _printBytes(uint8_t b[], int begin, int len) {
    int i;
    for (i=begin; i<begin+len; i++)
        printf("%x ", b[i]);
    //    cout << hex << b[i] << " " ;
    printf("\n");
}

int main(){
    int lambda = 127;
    int parallel = 1024000;
    int input_length = 64;
    int input_byte = ceil(input_length / 8);
    clock_t begin, end;
    begin = clock();
    aes_gen_block * cpu_aes_gen_block_array;
    aes_eval_block * cpu_aes_eval_block_array[2];
    cpu_aes_gen_block_array = new aes_gen_block[parallel];
    cpu_aes_eval_block_array[0] = new aes_eval_block[parallel];
    cpu_aes_eval_block_array[1] = new aes_eval_block[parallel];

    
    InputByteRelatedValuesGen cpu_values;
    cpu_values.r = (uint8_t*)malloc(parallel * input_byte * sizeof(uint8_t));
    // correction words, scw.shape = [parallel, input_length, input_byte]
    cpu_values.scw = (uint8_t*)malloc(parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t));
    cpu_values.tcw[0] = (bool*)malloc(parallel * input_length * sizeof(bool));
    cpu_values.tcw[1] = (bool*)malloc(parallel * input_length * sizeof(bool));
    cpu_values.output = (uint8_t*)malloc(parallel * input_byte * sizeof(uint8_t));
    // tcw.shape = [parallel, input_length]

    PRNG prng;
    bigint seed[2], r, res0, res1;
    prng.InitSeed();
    for(int i = 0; i < parallel; i++){
        // prng.get(r, input_length);
        r = 0;
        prng.get(seed[0], lambda);
        prng.get(seed[1], lambda);


        bytesFromBigint(&cpu_values.r[i * input_byte], r, input_byte);
        bytesFromBigint(&cpu_aes_eval_block_array[0][i].block[0], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_eval_block_array[0][i].block[LAMBDA_BYTE], seed[0], LAMBDA_BYTE);
        
        bytesFromBigint(&cpu_aes_eval_block_array[1][i].block[0], seed[1], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_eval_block_array[1][i].block[LAMBDA_BYTE], seed[1], LAMBDA_BYTE);
        
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[0][0], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[0][LAMBDA_BYTE], seed[0], LAMBDA_BYTE);

        bytesFromBigint(&cpu_aes_gen_block_array[i].block[1][0], seed[1], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[1][LAMBDA_BYTE], seed[1], LAMBDA_BYTE);
    }

    fss_dpf_generate(cpu_values, cpu_aes_gen_block_array, input_length, parallel);

    InputByteRelatedValuesEval cpu_eval_values_0;
    cpu_eval_values_0.result = (uint8_t *)malloc(parallel * input_byte * sizeof(uint8_t));
    fss_dpf_evaluate(cpu_eval_values_0, cpu_values, cpu_aes_eval_block_array[0], 0, input_length, parallel);
    
    InputByteRelatedValuesEval cpu_eval_values_1;
    cpu_eval_values_1.result = (uint8_t *)malloc(parallel * input_byte * sizeof(uint8_t));
    fss_dpf_evaluate(cpu_eval_values_1, cpu_values, cpu_aes_eval_block_array[1], 1, input_length, parallel);


    for(int i = 0; i < parallel; i++){
        bigintFromBytes(res0, &cpu_eval_values_0.result[i * input_byte], input_byte);
        bigintFromBytes(res1, &cpu_eval_values_1.result[i * input_byte], input_byte);
        if((res0 - res1)!=1){
            std::cout << res0 << std::endl;
            std::cout << res1 << std::endl;
            std::cout << "Error!" << res0 - res1 << std::endl;
        }
    }
    end = clock();
    std::cout << "Finished in " << double(end-begin) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;   

    for(int i = 0; i < parallel; i++){
        // prng.get(r, input_length);
        r = 6;  
        bytesFromBigint(&cpu_values.r[i * input_byte], r, input_byte);
    }

    // fss_dpf_generate_traverse(cpu_values, cpu_aes_gen_block_array, input_length, 8, parallel);

}   
