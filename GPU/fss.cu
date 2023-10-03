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
#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"

__global__ void fss_generate_gpu(fss_gen_struct * fss_gen, uint8_t * generated_value_0, uint8_t * key, uint8_t * generated_value_1, int numbytes){
    int bit_length = numbytes * 8, idx;
    for(int i = 0; i < bit_length - 1; i++){
        idx = int(i/8);
        memset(fss_gen -> keep, ((fss_gen -> r[idx]) >> (i - idx * 8))%2, 1); 
        memset(fss_gen -> lose, * fss_gen -> keep ^ 1, 1);
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                aes256_encrypt_ecb(fss_gen -> t[j], 1, fss_gen -> seed[0]);
            }
        }
    }
}

//uint8_t * seed_0, uint8_t * seed_1分别是初始化后的随机数种子
//uint8_t * generated_value_cpu_0是表示给party0生成的随机数结果存放位置， uint8_t * generated_value_cpu_1是表示给party1生成的随机数结果存放位置
void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * key, uint8_t * generated_value_cpu_0, uint8_t * generated_value_cpu_1, int numbytes){
    uint8_t * generated_value_0;
    uint8_t * generated_value_1;
    uint8_t *w_d;
    uint8_t *w;
    
    fss_gen_struct * fss_gen;
    int bit_length = numbytes * 8;
    //分配扩展密钥
    w = (uint8_t*)malloc(240*sizeof(uint8_t));
    aes_key_expansion(key, w);

    cudaMalloc((void**)&w_d, 240*sizeof(uint8_t));
    cudaMemcpy(w_d, w, 240*sizeof(uint8_t), cudaMemcpyHostToDevice);

    //分配输出的generated_value的长度
    cudaMalloc(&generated_value_0, (numbytes - 1) * 2 * (numbytes + 1) + numbytes);
    cudaMalloc(&generated_value_1, (numbytes - 1) * 2 * (numbytes + 1) + numbytes);
    //分配数据结构空间
    cudaMalloc(&fss_gen, sizeof(fss_gen_struct));
    cudaMalloc(&fss_gen->va, numbytes);
    cudaMalloc(&fss_gen->keep, 1);
    cudaMalloc(&fss_gen->lose, 1);
    for(int idx = 0 ; idx < 2 ; idx++){
        cudaMalloc(&fss_gen->seed[idx], numbytes);
        cudaMalloc(&fss_gen->s[idx], numbytes);
        cudaMalloc(&fss_gen->v[idx], numbytes);
        cudaMalloc(&fss_gen->t[idx], 1);
        cudaMalloc(&fss_gen->pre_t[idx], 1);
        cudaMalloc(&fss_gen->scw[idx], numbytes);
        cudaMalloc(&fss_gen->vcw[idx], numbytes);
        cudaMalloc(&fss_gen->tmp[idx], numbytes);
        cudaMalloc(&fss_gen->convert[idx], numbytes);
    }
    //初始化种子
    cudaMemcpy(fss_gen->r, r, numbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fss_gen->seed[0], seed0, numbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fss_gen->seed[1], seed1, numbytes, cudaMemcpyHostToDevice);
    cudaMemset(fss_gen->va, 0, cudaMemcpyHostToDevice);
    cudaMemset(fss_gen->pre_t[0], 0, cudaMemcpyHostToDevice);
    cudaMemset(fss_gen->pre_t[1], 1, cudaMemcpyHostToDevice);
    
    fss_generate_gpu(fss_gen, generated_value_cpu_0, generated_value_1, numbytes);
}

