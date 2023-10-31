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

__global__ void _mod2_t(uint8_t * org, bool dst, int fidx){
  dst = org[fidx] % 2;
}

__device__ void _add(uint8_t *a, uint8_t *b, uint8_t * res, int length){
  uint16_t tmp;
  bool carry = 0;
  uint16_t need_carry = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = a[i] + b[i] + carry;
    // printf("tmp is %d, a[i] is %d, b[i] is %d, carry is %d \n", tmp, a[i], b[i], carry);
    if(tmp < need_carry)
      carry = 0;
    else{
      carry = 1;
      tmp = tmp % need_carry;
    }
    res[i] = tmp;
  }
}

__device__ void _add(uint8_t *a, int value, int length){
  uint16_t tmp;
  bool carry = 0;
  uint16_t need_carry = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = a[i] + (value >> ((length - 1) * 8  - i * 8)) % 256 + carry;
    if(tmp < need_carry)
      carry = 0;
    else{
      carry = 1;
      tmp = tmp % need_carry;
    }
    a[i] = tmp;
  }
}

__device__ void _sub(uint8_t *minuend, int value, int length){
  int tmp;
  bool borrow = 0;
  uint16_t need_borrow = 1<<8;
  for(int i = length - 1; i >= 0; i--){
    tmp = minuend[i] - (value >> ((length - 1) * 8 - i * 8)) % 256 - borrow;
    if(tmp < 0){
      borrow = 1;
      tmp = need_borrow + tmp;
    }
    else{
      borrow = 0;
    }
    minuend[i] = tmp;
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
__device__ void _restricted_multiply(bool value, uint8_t * a, uint8_t * res, size_t num){
  for(int idx = 0; idx < num; idx++){
    res[idx] = value * a[idx];
  }
}

//求xor
__device__ void _xor(uint8_t * a, uint8_t * b, uint8_t * res, size_t num){
  for(int idx = 0; idx < num; idx++){
    res[idx] = b[idx] ^ a[idx];
  }
}

__device__ void _copy(uint8_t* org, uint8_t* dst, int org_begin, int dst_begin, size_t num) {
  for(int idx = 0; idx < num; idx++){
    dst[idx + dst_begin] = org[idx + org_begin];
  }
}

__global__ void printGpuBytes(uint8_t b[], int begin, int len) {
  for (int i=0; i<len; i++){
      // printf("%d\n",i);
      printf("%02x", b[i]);
  }
  //    cout << hex << b[i] << " " ;
  printf("\n");
}
#endif
