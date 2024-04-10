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

//

// __global__ void fss_generate_key_copy(FssGen * cuda_fss_gen, int numbytes){
//     printf("threadIdx.x is %d\n",threadIdx.x);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[threadIdx.x], cuda_fss_gen->seed[threadIdx.x], numbytes);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[threadIdx.x] + numbytes, cuda_fss_gen->seed[threadIdx.x], numbytes);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[threadIdx.x] + 2 * numbytes, cuda_fss_gen->seed[threadIdx.x], 1);
//     printGpuBytes<<<1,1>>>(cuda_fss_gen->inter_val[threadIdx.x], 2*numbytes+1);
// }

// __global__ void fss_gen_generate_random_values(FssGen * cuda_fss_gen, uint8_t * cuda_key[], int numbytes, int num_sm, int thrdperblock){
    
//     AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->inter_val[threadIdx.x], cuda_key[0], 176, numbytes, 3);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->t[threadIdx.x][0], cuda_fss_gen-> inter_val[threadIdx.x], 1);
//     _set_t<<<1,1>>>(1, cuda_fss_gen-> t[threadIdx.x][0], 1);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->v[threadIdx.x][0], cuda_fss_gen->inter_val[threadIdx.x] + 1, numbytes);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->s[threadIdx.x][0], cuda_fss_gen->inter_val[threadIdx.x] + 1 + numbytes, numbytes);

//     AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->inter_val[threadIdx.x], cuda_key[1], 176, numbytes, 3);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->t[threadIdx.x][1], cuda_fss_gen-> inter_val[threadIdx.x], 1);
//     _set_t<<<1,1>>>(1, cuda_fss_gen-> t[threadIdx.x][1], 1);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->v[threadIdx.x][1], cuda_fss_gen->inter_val[threadIdx.x] + 1, numbytes);
//     _copy<<<1,numbytes>>>(cuda_fss_gen->s[threadIdx.x][1], cuda_fss_gen->inter_val[threadIdx.x] + 1 + numbytes, numbytes);

// }

__global__ void vcw_generate_update_keep(FssGen * cuda_fss_gen, int numbytes){
    if(cuda_fss_gen->pre_t[1]){
          _add<<<1,1>>>(cuda_fss_gen->convert[0], cuda_fss_gen->va,  cuda_fss_gen->vcw, numbytes);
          _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->convert[1], cuda_fss_gen->vcw, numbytes);
    }
    else{
        _sub<<<1,1>>>(cuda_fss_gen->convert[1], cuda_fss_gen->convert[0], cuda_fss_gen->vcw, numbytes);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->va, cuda_fss_gen->vcw, numbytes);
    }

    if(cuda_fss_gen->pre_t[1]){
        _sub_1<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->vcw, numbytes);
    }
    else{
        _add_c<<<1,1>>>(cuda_fss_gen->vcw, 1, cuda_fss_gen->vcw, numbytes);
    }   
}


__global__ void vcw_generate_update_lose(FssGen * cuda_fss_gen, int numbytes){
    if(cuda_fss_gen->pre_t[1]){
          _add<<<1,1>>>(cuda_fss_gen->convert[0], cuda_fss_gen->va,  cuda_fss_gen->vcw, numbytes);
          _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->convert[1], cuda_fss_gen->vcw, numbytes);
    }
    else{
        _sub<<<1,1>>>(cuda_fss_gen->convert[1], cuda_fss_gen->convert[0], cuda_fss_gen->vcw, numbytes);
        _sub<<<1,1>>>(cuda_fss_gen->vcw, cuda_fss_gen->va, cuda_fss_gen->vcw, numbytes);
    }
}

__global__ void va_genearte_update(FssGen * cuda_fss_gen, int keep, int numbytes){
    _sub<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->v[keep][1], cuda_fss_gen->va, numbytes);
    _add<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->v[keep][0], cuda_fss_gen->va, numbytes);
    if(cuda_fss_gen->pre_t[1]){
        _sub<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->vcw, cuda_fss_gen->va, numbytes);
    }
    else{
        _add<<<1,1>>>(cuda_fss_gen->va, cuda_fss_gen->vcw, cuda_fss_gen->va, numbytes);
    }
}

__global__ void tcw_pret_generate_update(FssGen * cuda_fss_gen, int keep, int numbytes){
    for(int j = 0; j < 2; j++){
        cuda_fss_gen->tcw[j][0] = cuda_fss_gen->t[j][0][0] ^ cuda_fss_gen->t[j][0][1] ^ keep ^ 1;
        _restricted_multiply<<<1,numbytes>>>(cuda_fss_gen->pre_t[j], cuda_fss_gen->scw, cuda_fss_gen->seed[j], numbytes);
        _xor<<<1,numbytes>>>(cuda_fss_gen->s[keep][j], cuda_fss_gen->seed[j], cuda_fss_gen->seed[j], numbytes);
        cuda_fss_gen->pre_t[j] = cuda_fss_gen->t[keep][j][0] ^ (cuda_fss_gen->pre_t[j] * cuda_fss_gen->tcw[keep][0]);
    }
}

__global__ void final_cw_generate_update(FssGen * cuda_fss_gen, uint8_t * generated_value, int bit_length, int numbytes){
    if(cuda_fss_gen->pre_t[1]){
        _sub<<<1,1>>>(cuda_fss_gen->convert[1], cuda_fss_gen->convert[0], generated_value + (bit_length - 1) * 2 * (numbytes + 1), numbytes);
        _sub<<<1,1>>>(generated_value + (bit_length - 1) * 2 * (numbytes + 1), cuda_fss_gen->va, generated_value + (bit_length - 1) * 2 * (numbytes + 1), numbytes);
    }
    else{
        _add<<<1,1>>>(cuda_fss_gen->convert[0], cuda_fss_gen->va, generated_value + (bit_length - 1) * 2 * (numbytes + 1), numbytes);
        _sub<<<1,1>>>(generated_value + (bit_length - 1) * 2 * (numbytes + 1), cuda_fss_gen->convert[1], generated_value + (bit_length - 1) * 2 * (numbytes + 1), numbytes);
    }
}


__global__ void correction_word_eval_copy(FssEval * cuda_fss_eval, uint8_t * generated_value, int numbytes, int idx){
    _copy<<<1,numbytes>>>(generated_value, cuda_fss_eval->scw, 2*(numbytes+1)*idx, 0, numbytes);
    _copy<<<1,numbytes>>>(generated_value, cuda_fss_eval->vcw, 2*(numbytes+1)*idx + numbytes, 0, numbytes); 
    _copy<<<1,numbytes>>>(generated_value, cuda_fss_eval->tcw[0], 2*(numbytes+1)*idx + 2 * numbytes, 0, 1);
    _copy<<<1,numbytes>>>(generated_value, cuda_fss_eval->tcw[1], 2*(numbytes+1)*idx + 2 * numbytes + 1, 0, 1);
    _restricted_multiply<<<1,numbytes>>>(cuda_fss_eval->pre_t, cuda_fss_eval->scw, cuda_fss_eval->s[idx], numbytes);
    _restricted_multiply<<<1,numbytes>>>(cuda_fss_eval->pre_t, cuda_fss_eval->tcw[idx], cuda_fss_eval->t[idx], numbytes);
    _xor<<<1,numbytes>>>(cuda_fss_eval->s[idx], cuda_fss_eval->s_hat[idx], cuda_fss_eval->s[idx], numbytes);
    _xor<<<1,numbytes>>>(cuda_fss_eval->t[idx], cuda_fss_eval->t_hat[idx], cuda_fss_eval->t[idx], numbytes);
}

__global__ void random_value_eval_generate(FssEval * cuda_fss_eval, uint8_t * cuda_key, int numbytes, int num_sm, int thrdperblock){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    _copy<<<1,numbytes>>>(cuda_fss_eval->seed, cuda_fss_eval->inter_val, 0, 0, numbytes);
    _copy<<<1,numbytes>>>(cuda_fss_eval->seed, cuda_fss_eval->inter_val, 0, numbytes, numbytes);
    _copy<<<1,numbytes>>>(cuda_fss_eval->seed, cuda_fss_eval->inter_val, 0, 2*numbytes, 1);
    AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_eval->inter_val, cuda_key, 176, numbytes, 3);
    _copy<<<1,numbytes>>>(cuda_fss_eval->inter_val, cuda_fss_eval->t_hat[idx], 0, 0, 1);
    _mod2_t<<<1,1>>>(cuda_fss_eval->t_hat[idx]);
    _copy<<<1,numbytes>>>(cuda_fss_eval->inter_val, cuda_fss_eval->v_hat[idx], 1, 0, numbytes);
    _copy<<<1,numbytes>>>(cuda_fss_eval->inter_val, cuda_fss_eval->s_hat[idx], numbytes + 1, 0,  numbytes);
}

__global__ void convert_random_value_eval_generate(FssEval * cuda_fss_eval, uint8_t * cuda_key, int numbytes, int num_sm, int thrdperblock){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    _copy<<<1,numbytes>>>(cuda_fss_eval->v_hat[idx], cuda_fss_eval->convert_seed, 0, 0, numbytes);
    AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_eval->convert_seed, cuda_key, 176, numbytes, 1);
    _copy<<<1,numbytes>>>(cuda_fss_eval->convert_seed, cuda_fss_eval->v_hat[idx], 0, 0, numbytes);
}

__global__ void value_eval_update(FssEval * cuda_fss_eval, int cur_bit, int party, int numbytes){
    _restricted_multiply<<<1, numbytes>>>(cuda_fss_eval->pre_t, cuda_fss_eval->vcw, cuda_fss_eval->vcw, numbytes);
    _add<<<1, 1>>>(cuda_fss_eval->vcw, cuda_fss_eval->convert[cur_bit],cuda_fss_eval->convert[cur_bit], numbytes);
    if(party){
        _sub<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[cur_bit],cuda_fss_eval->tmp_v, numbytes);
    }
    else{
        _add<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert[cur_bit],cuda_fss_eval->tmp_v, numbytes);
    }
    _copy<<<1, numbytes>>>(cuda_fss_eval->s[cur_bit], cuda_fss_eval->seed, 0, 0, numbytes);
    cuda_fss_eval->pre_t = cuda_fss_eval->t[cur_bit][0]%2;
}

__global__ void final_eval_update(FssEval * cuda_fss_eval, uint8_t * cuda_key, uint8_t * generated_value, int party, int numbytes, int num_sm, int thrdperblock){
    
    _copy<<<1,numbytes>>>(cuda_fss_eval->seed, cuda_fss_eval->convert_seed, 0, 0, numbytes);
    AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_eval->convert_seed, cuda_key, 176, numbytes, 1);

    _restricted_multiply<<<1, numbytes>>>(cuda_fss_eval->pre_t, cuda_fss_eval->vcw, cuda_fss_eval->vcw, numbytes);
    _add<<<1, 1>>>(cuda_fss_eval->vcw, cuda_fss_eval->convert_seed,cuda_fss_eval->convert_seed, numbytes);
    if(party){
        _sub<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert_seed, cuda_fss_eval->tmp_v, numbytes);
    }
    else{
        _add<<<1,1>>>(cuda_fss_eval->tmp_v, cuda_fss_eval->convert_seed, cuda_fss_eval->tmp_v, numbytes);
    }
}

#endif