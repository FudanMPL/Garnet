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

__global__ void fss_generate_gpu_64(fss_gen_struct_64 * fss_gen, uint8_t * key, uint8_t * generated_value_0, uint8_t * generated_value_1,  int numbytes){
    int bit_length = numbytes * 8, idx, inter_int_value;
    for(int i = 0; i < bit_length - 1; i++){
        idx = int(i/8);
        inter_int_value = ((fss_gen -> r[idx]) >> (i - idx * 8))%2;
        fss_gen -> keep = inter_int_value; 
        fss_gen -> lose = inter_int_value ^ 1; 
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                printf("fss_generating random values");
                copyKernel(fss_gen -> inter_val[j], fss_gen -> seed[j], numbytes);
                copyKernel(fss_gen -> inter_val[j] + numbytes, fss_gen -> seed[j], numbytes);
                copyKernel(fss_gen -> inter_val[j] + 2*numbytes, fss_gen -> seed[j], 1);
                
                aes256Kernel(fss_gen -> inter_val[j], 2 * numbytes + 1, key);
                
                copyKernel(fss_gen -> t[j][k], fss_gen -> inter_val[j], 1);
                fss_gen -> t[j][k][0] = fss_gen -> t[j][k][0] % 2;
                copyKernel(fss_gen -> v[j][k], fss_gen -> inter_val[j] + 1, numbytes);
                copyKernel(fss_gen -> s[j][k], fss_gen -> inter_val[j] + 1 + numbytes, numbytes);
                
            }
        }
        
    }
}

__global__ void fss_generate_gpu_128(fss_gen_struct_128 * fss_gen, uint8_t * key, uint8_t * generated_value_0, uint8_t * generated_value_1,  int numbytes){
    int bit_length = numbytes * 8, idx, inter_int_value;
    for(int i = 0; i < bit_length - 1; i++){
        idx = int(i/8);
        inter_int_value = ((fss_gen -> r[idx]) >> (i - idx * 8))%2;
        fss_gen -> keep = inter_int_value; 
        fss_gen -> lose = inter_int_value ^ 1; 
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                printf("fss_generating random values");
                copyKernel(fss_gen -> inter_val[j], fss_gen -> seed[j], numbytes);
                copyKernel(fss_gen -> inter_val[j] + numbytes, fss_gen -> seed[j], numbytes);
                copyKernel(fss_gen -> inter_val[j] + 2*numbytes, fss_gen -> seed[j], 1);
                
                aes256Kernel(fss_gen -> inter_val[j], 2 * numbytes + 1, key);
                
                copyKernel(fss_gen -> t[j][k], fss_gen -> inter_val[j], 1);
                fss_gen -> t[j][k][0] = fss_gen -> t[j][k][0] % 2;
                copyKernel(fss_gen -> v[j][k], fss_gen -> inter_val[j] + 1, numbytes);
                copyKernel(fss_gen -> s[j][k], fss_gen -> inter_val[j] + 1 + numbytes, numbytes);
                
            }
        }
        
    }
}

#endif