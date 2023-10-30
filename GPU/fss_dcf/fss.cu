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

#ifndef FSS_CU_
#define FSS_CU_

__global__ void rand_svt_generate(FssGen * cuda_fss_gen, uint8_t * cuda_key, int j, int k, int num_sm, int thrdperblock){

    _copy<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->seed[j], cuda_fss_gen->inter_val[j], 0, 0, BYTE_LAMBDA);
    _copy<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->seed[j], cuda_fss_gen->inter_val[j], 0, BYTE_LAMBDA, BYTE_LAMBDA);  
    AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->inter_val[j], cuda_key, 176, BYTE_LAMBDA, 2);
    rshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->v[k][j], 0, 0, BYTE_LAMBDA);
    rshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->s[k][j], BYTE_LAMBDA, 0,  BYTE_LAMBDA);
    cuda_fss_gen->t[k][j] = cuda_fss_gen->s[k][j][BYTE_LAMBDA-1] % 2;
}

__global__ void vcw_generate_update_pre_t(FssGen * cuda_fss_gen, int keep){  
    if(cuda_fss_gen->pre_t[1]){
        _add<<<1,1>>>(cuda_fss_gen->convert[0], cuda_fss_gen->va,  cuda_fss_gen->vcw, BYTE_LEN);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->convert[1], cuda_fss_gen->vcw, BYTE_LEN);
    }
    else{
        _sub<<<1,1>>>(cuda_fss_gen->convert[1], cuda_fss_gen->convert[0], cuda_fss_gen->vcw,  BYTE_LEN);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->va, cuda_fss_gen->vcw,  BYTE_LEN);
    }
}

__global__ void vcw_generate_update_keep(FssGen * cuda_fss_gen){
    if(cuda_fss_gen->pre_t[1]){
        _sub<<<1,1>>>(cuda_fss_gen->vcw, 1, BYTE_LAMBDA);
    }
    else{
        _add<<<1,1>>>(cuda_fss_gen->vcw, 1,  BYTE_LAMBDA);
    }   
}


__global__ void va_genearte_update(FssGen * cuda_fss_gen, int keep){
    _sub<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->convert[1], cuda_fss_gen->va, BYTE_LEN);
    _add<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->convert[0], cuda_fss_gen->va, BYTE_LEN);

    if(cuda_fss_gen->pre_t[1]){
        _sub<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->vcw, cuda_fss_gen->va, BYTE_LEN);
    }
    else{
        _add<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->vcw, cuda_fss_gen->va, BYTE_LEN);
    }
}

__global__ void stcw_pret_generate_update(FssGen * cuda_fss_gen, int keep){
    cuda_fss_gen->tcw[0] = cuda_fss_gen->t[0][0] ^ cuda_fss_gen->t[0][1] ^ keep ^ 1;
    cuda_fss_gen->tcw[1] = cuda_fss_gen->t[1][0] ^ cuda_fss_gen->t[1][1] ^ keep;
    for(int j = 0; j < 2; j++){
        _restricted_multiply<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->pre_t[j], cuda_fss_gen->scw, cuda_fss_gen->seed[j], BYTE_LAMBDA); 
        _xor<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->s[keep][j], cuda_fss_gen->seed[j], cuda_fss_gen->seed[j],  BYTE_LAMBDA);
        cuda_fss_gen->pre_t[j] = cuda_fss_gen->t[keep][j] ^ (cuda_fss_gen->pre_t[j] * cuda_fss_gen->tcw[keep]);    
    }
}

__global__ void generated_value_update(FssGen * cuda_fss_gen, uint8_t * generated_value,  int idx){
    lshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->scw, generated_value, 0, 2*BYTE_LAMBDA*idx, BYTE_LAMBDA);
    generated_value[2*BYTE_LAMBDA*idx + BYTE_LAMBDA - 1] += cuda_fss_gen->tcw[0];
    lshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->vcw, generated_value, 0, 2*BYTE_LAMBDA*idx + BYTE_LAMBDA, BYTE_LAMBDA); 
    generated_value[2*BYTE_LAMBDA*idx + 2 * BYTE_LAMBDA - 1] += cuda_fss_gen->tcw[1];
}

__global__ void final_cw_generate_update(FssGen * cuda_fss_gen, uint8_t * generated_value, int bit_length, int numbytes){
    if(cuda_fss_gen->pre_t[1]){
        _add<<<1,1>>>(cuda_fss_gen->convert[0], cuda_fss_gen->va, cuda_fss_gen->vcw, numbytes);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->convert[1],  cuda_fss_gen->vcw, numbytes);
    }
    else{
        _sub<<<1,1>>>(cuda_fss_gen->convert[1], cuda_fss_gen->convert[0], cuda_fss_gen->vcw, numbytes);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->va, cuda_fss_gen->vcw, numbytes);
    }
    _copy<<<1,numbytes>>>(cuda_fss_gen->vcw, generated_value, 0, (bit_length) * 2 * (numbytes + 1), numbytes);
}

__global__ void cw_eval_copy(FssEval * cuda_fss_eval, uint8_t * generated_value, int i){
    rshift_1<<<1,BYTE_LAMBDA>>>(generated_value, cuda_fss_eval->scw, 2*BYTE_LAMBDA*i, 0, BYTE_LAMBDA);
    rshift_1<<<1,BYTE_LAMBDA>>>(generated_value, cuda_fss_eval->vcw, 2*BYTE_LAMBDA*i + BYTE_LAMBDA, 0, BYTE_LAMBDA); 
    cuda_fss_eval->tcw[0] = generated_value[2*BYTE_LAMBDA*i + BYTE_LAMBDA - 1]%2;
    cuda_fss_eval->tcw[0] = generated_value[2*BYTE_LAMBDA*i + 2 * BYTE_LAMBDA - 1]%2;
}

__global__ void rand_svt_eval_generate(FssEval * cuda_fss_eval, uint8_t * cuda_key, int j, int num_sm, int thrdperblock){

    _copy<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->seed, cuda_fss_eval->inter_val[j], 0, 0, BYTE_LAMBDA);
    _copy<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->seed, cuda_fss_eval->inter_val[j], 0, BYTE_LAMBDA, BYTE_LAMBDA);  
    AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_eval->inter_val[j], cuda_key, 176, BYTE_LAMBDA, 2);
    rshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->inter_val[j], cuda_fss_eval->v_hat[j], 0, 0, BYTE_LAMBDA);
    rshift_1<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->inter_val[j], cuda_fss_eval->s_hat[j], BYTE_LAMBDA, 0,  BYTE_LAMBDA);
    cuda_fss_eval->t_hat[j] = cuda_fss_eval->s_hat[j][BYTE_LAMBDA-1] % 2;
}

__global__ void st_eval_phrase(FssEval * cuda_fss_eval){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    _restricted_multiply<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->pre_t, cuda_fss_eval->scw, cuda_fss_eval->s[idx], BYTE_LAMBDA);
    _xor<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->s[idx], cuda_fss_eval->s_hat[idx], cuda_fss_eval->s[idx], BYTE_LAMBDA);
    cuda_fss_eval->t[idx] = (cuda_fss_eval->tcw[idx] * cuda_fss_eval->pre_t) ^ cuda_fss_eval->t_hat[idx];
}

__global__ void value_eval_update(FssEval * cuda_fss_eval, uint8_t * cuda_key, int cur_bit, int party, int num_sm, int thrdperblock){
    _copy<<<1,BYTE_LEN>>>(cuda_fss_eval->v_hat[cur_bit], cuda_fss_eval->convert[cur_bit], 0, 0, BYTE_LEN);
    _restricted_multiply<<<1,BYTE_LAMBDA>>>(cuda_fss_eval->pre_t, cuda_fss_eval->vcw, cuda_fss_eval->vcw, BYTE_LAMBDA);
    _add<<<1,1>>>(cuda_fss_eval->convert[cur_bit], cuda_fss_eval->vcw, cuda_fss_eval->convert[cur_bit],BYTE_LEN);
    if(party){
        _sub<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[cur_bit],cuda_fss_eval->tmp_v, BYTE_LEN);
    }
    else{
        _add<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[cur_bit],cuda_fss_eval->tmp_v, BYTE_LEN);
    }
    _copy<<<1, BYTE_LAMBDA>>>(cuda_fss_eval->s[cur_bit], cuda_fss_eval->seed, 0, 0, BYTE_LAMBDA);
    cuda_fss_eval->pre_t = cuda_fss_eval->t[cur_bit]%2;
}

__global__ void final_eval_update(FssEval * cuda_fss_eval, uint8_t * cuda_key, uint8_t * generated_value, uint8_t * cuda_result, int party, int num_sm, int thrdperblock){
    _copy<<<1,BYTE_LEN>>>(cuda_fss_eval->seed, cuda_fss_eval->convert[0], 0, 0, BYTE_LEN);
    _restricted_multiply<<<1, BYTE_LEN>>>(cuda_fss_eval->pre_t, generated_value + (BYTE_LEN * 8) * 2 * BYTE_LAMBDA, generated_value + (BYTE_LEN * 8) * 2 * BYTE_LAMBDA, BYTE_LEN);
    _add<<<1, 1>>>(generated_value + (BYTE_LEN * 8) * 2 * BYTE_LAMBDA, cuda_fss_eval->convert[0], cuda_fss_eval->convert[0], BYTE_LAMBDA);
    if(party){
        _sub<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[0], cuda_result, BYTE_LEN);
    }
    else{
        _add<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[0], cuda_result, BYTE_LEN);
    }
}

#endif