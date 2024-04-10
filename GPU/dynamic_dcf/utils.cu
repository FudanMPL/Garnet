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

__device__ void _mod2_t(uint8_t * a){
  a[0] = a[0] % 2;
}

__device__ void _set(uint8_t * org, uint8_t * dest, size_t num){
  for(int idx = 0; idx < num; idx++){
    dest[idx] = org[idx];
  }
}

__device__ void _add(uint8_t *a, uint8_t *b, uint8_t * res, int length){
  uint16_t tmp;
  bool carry = 0;
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

__device__ void _add_c(uint8_t *a, int constant, uint8_t * res, int length){
  uint16_t tmp;
  bool carry = 0;
  uint16_t need_carry = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = a[i] + ((constant >> (i * 8))%256) + carry;
    if(tmp < need_carry)
      carry = 0;
    else{
      carry = 1;
      tmp = tmp % need_carry;
    }
    res[i] = tmp;
  }
}

__device__ void _sub_1(uint8_t *minuend, uint8_t * res, int length){
  int tmp;
  bool borrow = 0;
  uint16_t need_borrow = 1<<8;
  
  tmp = minuend[0] - 1;
  if(tmp < 0){
    borrow = 1;
    tmp = need_borrow + tmp;
  }
  else{
    borrow = 0;
  }
  res[0] = tmp;

  if(borrow){
    for(int i = length - 2; i >= 0; i--){
      tmp = minuend[i] - borrow;
      if(tmp < 0){
        borrow = 1;
        tmp = need_borrow + tmp;
      }
      else{
        borrow = 0;
        break;
      }
      res[i] = tmp;
    }  
  }
}

__device__ void _sub(uint8_t *minuend, uint8_t *subtrahend, uint8_t * res, int length){
  int tmp;
  bool borrow = 0;
  uint16_t need_borrow = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = minuend[i] - subtrahend[i] - borrow;
    if(tmp < 0){
      borrow = 1;
      tmp = need_borrow + tmp;
    }
    else{
      borrow = 0;
    }
    res[i] = tmp;
  }  
}

//我们要求第一个数是0/1
__device__ void _restricted_multiply(int value, uint8_t * a, uint8_t * res, size_t num){
  for(int idx = 0; idx < num; idx++){
    res[idx] = value * a[idx];
  }
}

//求xor
__device__ void _xor(uint8_t * a, uint8_t * b, uint8_t * res, size_t num){
  for(int idx = 0; idx < num; idx++) {
    res[idx] = b[idx] ^ a[idx];
  }
}

__device__ void _copy(uint8_t* org, uint8_t* dst, int org_begin, int dst_begin, size_t num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int idx = 0; idx < num; idx++){
    dst[idx + dst_begin] = org[idx + org_begin];
  }
}

__device__ void resMulKernel(int value, uint8_t * a, uint8_t * res, size_t num){
  _restricted_multiply(value, a, res, num);
}

__device__ void subKernel(uint8_t * minus, uint8_t * sub, uint8_t * res, int length){
  _sub(minus, sub, res, length);
  return;
}

__device__ void addKernel(uint8_t * a, uint8_t * b, uint8_t * res, int length){
  _add(a, b, res, length);
  return;
}

__device__ void addCKernel(uint8_t * a, int constant, uint8_t * res, int length){
  _add_c(a, constant, res, length);
  return;
}



__device__ void xorKernel(uint8_t * a, uint8_t * b, uint8_t * res, size_t num){
  _xor(a, b, res, num);
  return;
}

__global__ void printGpuBytes(uint8_t b[], int len) {
int i;
for (i=0; i<len; i++)
    printf("%x ", b[i]);
//    cout << hex << b[i] << " " ;
printf("\n----------------------\n");
}
#endif
