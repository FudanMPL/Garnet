/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-24 16:24:02
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-30 21:39:20
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
    int create_size;
    if(parallel % MAX_PARALLEL > 0){
        create_size = int(parallel / MAX_PARALLEL) + 1;
    }
    else{
        create_size = int(parallel / MAX_PARALLEL);
    }
    
    std::cout << create_size << std::endl;
    FssDpfGenerateBlock * cpu_gen_block = new FssDpfGenerateBlock[create_size];
    
    PRNG prng;
    bigint seed[2], r;
    prng.InitSeed();
    int group;
    for(int i = 0; i < parallel; i++){
        group = int(i / MAX_PARALLEL);
        prng.get(r, lambda);
        // prng.get(seed[0], lambda);
        // prng.get(seed[1], lambda);
        seed[0] = 120390;
        seed[1] = 129832;
        bytesFromBigint(&cpu_gen_block[group].seed[0][(i - group * MAX_PARALLEL) * LAMBDA_BYTE], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&cpu_gen_block[group].seed[1][(i - group * MAX_PARALLEL) * LAMBDA_BYTE], seed[1], LAMBDA_BYTE);
    }
    CorrectionWord * cw_cpu = new CorrectionWord[create_size];
    fss_dpf_generate(cpu_gen_block, cw_cpu, parallel, create_size);

}