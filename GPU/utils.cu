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


#ifndef UTILS_CU_
#define UTILS_CU_

__global__ void _set(int value, uint8_t * a, size_t num){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    a[idx] = (value >> idx) % 256;
  }
}

__global__ void _add(uint8_t *a, uint8_t *b, uint8_t * res, int length){
  uint16_t tmp, carry = 0;
  uint16_t need_carry = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = a[i] + b[i] + carry;
    if(tmp < need_carry)
      carry = 0;
    else{
      carry = 1;
      tmp = tmp % need_carry;
    }
    res[i] = tmp;
  }
}

//我们要求第一个数是0/1
__global__ void _restricted_multiply(int *value, uint8_t * a, uint8_t * res, size_t num){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    res[idx] = *value * a[idx];
  }
}

//求xor
__global__ void _xor(uint8_t * a, uint8_t * b, uint8_t * res, size_t num){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    res[idx] = b[idx] ^ a[idx];
  }
}

__global__ void _copy(uint8_t* a, uint8_t* b, size_t num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    b[idx] = a[idx];
  }
}

__device__ void copyKernel(uint8_t* a, uint8_t* b, size_t num){
  _copy<<<1,num>>>(a, b, num);
  return;
}

__device__ void xorKernel(uint8_t * a, uint8_t * b, uint8_t * res, size_t num){
  _xor<<<1,num>>>(a, b, res, num);
  return;
}

__device__ void setKernel(int value, uint8_t * a, size_t num){
  _set<<<1,num>>>(value, a, num);
}
#endif
