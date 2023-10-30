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

#ifndef FSS_STRUCT_H_
#define FSS_STRUCT_H_

#define INPUT_BYTE 8
#define LAMBDA_BYTE 16
#define MAX_PARALLEL 1024

class aes_block
{
public:
    uint8_t block[LAMBDA_BYTE];
};

class KeyBlock{
public:
    uint8_t cuda_key[2][240];
};

class FssDpfGenerateBlock
{
public:
    uint8_t r[INPUT_BYTE * MAX_PARALLEL];
    uint8_t seed[2][LAMBDA_BYTE * MAX_PARALLEL];
};

class CorrectionWord{
public:
    uint8_t scw[INPUT_BYTE * 8 * MAX_PARALLEL][LAMBDA_BYTE];
    bool tcw[2 * MAX_PARALLEL];
    uint8_t output[LAMBDA_BYTE * MAX_PARALLEL];
};

class CorrectionWordCompress8{
public:
    uint8_t scw[INPUT_BYTE * 8 - 3][LAMBDA_BYTE];
    bool tcw[2];
    uint8_t output[LAMBDA_BYTE];
};

class FssDpfGen
{
public:
    uint8_t s[2][2][LAMBDA_BYTE * MAX_PARALLEL];
    bool t[2][2][MAX_PARALLEL];
    bool pre_t[2][MAX_PARALLEL]; 
    bool keep[MAX_PARALLEL];
    bool lose[MAX_PARALLEL];
};

class FssDpfEval
{
public:
    uint8_t s[2][2][LAMBDA_BYTE * MAX_PARALLEL];
    bool t[2][2][MAX_PARALLEL];
    bool pre_t[2][MAX_PARALLEL]; 
};


#endif