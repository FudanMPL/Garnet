/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-24 16:24:02
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-31 14:06:48
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
#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTE 16
#define INPUT_BYTE 8


int main(){
    int lambda = 127;
    int bit_length = INPUT_BYTE * 8;
    int parallel = 1024;
    
    RandomValueBlock * cpu_r_block = new RandomValueBlock[parallel];
    aes_block * cpu_aes_block_array[2];
    cpu_aes_block_array[0] = new aes_block[parallel];
    cpu_aes_block_array[1] = new aes_block[parallel];
    CorrectionWord * cw_cpu = new CorrectionWord[parallel];
    
    PRNG prng;
    bigint seed[2], r;
    prng.InitSeed();
    int group;
    for(int i = 0; i < parallel; i++){
        // prng.get(r, lambda);
        r = 12094302;
        // prng.get(seed[0], lambda);
        // prng.get(seed[1], lambda);
        seed[0] = 120390;
        seed[1] = 129832;
        bytesFromBigint(&cpu_r_block[i].r[0], r, LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_block_array[0][i].block[0], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_block_array[1][i].block[0], seed[1], LAMBDA_BYTE);
    }
    // for(int i = 0; i < 100; i++){
    //     bigintFromBytes(r, &cpu_r_block[i].r[0], LAMBDA_BYTE);
    //     std::cout << r << std::endl;
    // }

    fss_dpf_generate(cpu_r_block, cpu_aes_block_array, cw_cpu, parallel);

}