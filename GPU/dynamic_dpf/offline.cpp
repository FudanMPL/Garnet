#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include "gpu.h"
#include "Math/bigint.h"
#include "Protocols/Fss3Prep.h"
#include "gpu.h"
#include <ctime>
#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTE 16

void _printBytes(uint8_t b[], int begin, int len) {
    int i;
    for (i=begin; i<begin+len; i++)
        printf("%x ", b[i]);
    //    cout << hex << b[i] << " " ;
    printf("\n");
}

int main(int argc, const char** argv){
    int lambda = 127;
    int parallel = 1024;
    int input_length = 64;
    int input_byte = ceil(input_length / 8);
    int party = 0;
    string file_name = "";
    ez::ezOptionParser opt;
    fstream f[2];

    opt.add(
        "64", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "bit length", // Help description.
        "-bl", // Flag token.
        "--bit_length" // Flag token.
    );
    opt.add(
        "1024", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "batch_size", // Help description.
        "-b", // Flag token.
        "--batch_size" // Flag token.
    );
    opt.add(
        "Player-Data/2-fss/dpf_correction_word", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "file name", // Help description.
        "-fn", // Flag token.
        "--file_name" // Flag token.
    );
    opt.parse(argc, argv); 
    opt.get("-bl")->getInt(input_length);
    opt.get("-b")->getInt(parallel);
    opt.get("-fn")->getString(file_name);
    file_name = file_name + "_" + std::to_string(input_length) + "_" + std::to_string(parallel) + "_";
    std::cout << party << " " << input_length << " " << parallel << std::endl;

    clock_t begin, end;
    begin = clock();
    aes_gen_block * cpu_aes_gen_block_array;
    aes_eval_block * cpu_aes_eval_block_array[2];
    cpu_aes_gen_block_array = new aes_gen_block[parallel];
    cpu_aes_eval_block_array[0] = new aes_eval_block[parallel];
    cpu_aes_eval_block_array[1] = new aes_eval_block[parallel];

    InputByteRelatedValuesGen cpu_values;
    cpu_values.r = (uint8_t*)malloc(parallel * input_byte * sizeof(uint8_t));
    // correction words, scw.shape = [parallel, input_length, input_byte]
    cpu_values.scw = (uint8_t*)malloc(parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t));
    cpu_values.tcw[0] = (bool*)malloc(parallel * input_length * sizeof(bool));
    cpu_values.tcw[1] = (bool*)malloc(parallel * input_length * sizeof(bool));
    cpu_values.output = (uint8_t*)malloc(parallel * input_byte * sizeof(uint8_t));
    // tcw.shape = [parallel, input_length]

    PRNG prng;
    bigint seed[2][parallel], r[parallel], r_share[2][parallel], res0, res1;
    prng.InitSeed();
    f[0].open(file_name+"0", ios::out);
    f[1].open(file_name+"1", ios::out);
    
    for(int i = 0; i < parallel; i++){
        prng.get(r_share[0][i], input_length - 1);
        prng.get(r_share[1][i], input_length - 1);
        r[i] = r_share[0][i] + r_share[1][i];
        prng.get(seed[0][i], input_length);
        prng.get(seed[1][i], input_length);
        bytesFromBigint(&cpu_values.r[i * input_byte], r[i], input_byte);
        bytesFromBigint(&cpu_aes_eval_block_array[0][i].block[0], seed[0][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_eval_block_array[0][i].block[LAMBDA_BYTE], seed[0][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_eval_block_array[1][i].block[0], seed[1][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_eval_block_array[1][i].block[LAMBDA_BYTE], seed[1][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[0][0], seed[0][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[0][LAMBDA_BYTE], seed[0][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[1][0], seed[1][i], LAMBDA_BYTE);
        bytesFromBigint(&cpu_aes_gen_block_array[i].block[1][LAMBDA_BYTE], seed[1][i], LAMBDA_BYTE);
    }

    fss_dpf_generate(cpu_values, cpu_aes_gen_block_array, input_length, parallel);
    bigint scw, output;
    // r_share
    std::cout << file_name << std::endl;
    // seed
    std::cout << seed[party] << std::endl;

    for(int p = 0; p < 2; p++){
        for(int j = 0; j < parallel; j++){
            f[p] << r_share[p][j] << std::endl;
            f[p] << seed[p][j] << std::endl;
            for(int i = j * input_length; i < (j+1) * input_length; i++){
                bigintFromBytes(scw, &cpu_values.scw[i * LAMBDA_BYTE], LAMBDA_BYTE);
                f[p] << scw << std::endl;
                f[p] << cpu_values.tcw[0][i] << std::endl;
                f[p] << cpu_values.tcw[1][i] << std::endl;
            }
            bigintFromBytes(output, &cpu_values.output[j * input_byte], input_byte);
            f[p] << output << std::endl;
        }
    }
    end = clock();
}   
