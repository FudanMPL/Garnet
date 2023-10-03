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

__global__ void add(uint8_t *a, uint8_t *b, uint8_t * res, int length){
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
__global__ void restricted_multiply(int *value, uint8_t * a, uint8_t * res, int length){
  if(value[0] == 1){
    memcpy(res, a, sizeof(res));
  }
  else{
    memset(res, 0, sizeof(res));
  }
}