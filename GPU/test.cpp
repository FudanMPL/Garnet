/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-03 00:05:44
 * @FilePath: /txy/Garnet/GPU/test.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include "gpu.h"
#include "Math/bigint.h"

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512
#define LAMBDA_BYTES 8
int main(int argc,char** argv){


    FILE *file;
    std::fstream k_in;
    uint8_t *buf; 
    char *fname;
    int  i;
    int padding;
    int lambda_bytes = LAMBDA_BYTES;


    
    bigint tmp[2], s[2][2], v[2][2],  t[2][2];
    octet seed[2][2*(2*LAMBDA_BYTES+1)], result[LAMBDA_BYTES], a[LAMBDA_BYTES], b[LAMBDA_BYTES];
    uint8_t res[LAMBDA_BYTES];
    // generate key
    tmp[0] = 6213237549190506053;
    tmp[1] = 8248733145573590288;
    bytesFromBigint(&a[0], tmp[0], LAMBDA_BYTES);
    bytesFromBigint(&b[0], tmp[1], LAMBDA_BYTES);
    test_add(a, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[0] + tmp[1] << std::endl;

    test_restricted_multiply(1, b, result, LAMBDA_BYTES);
    bigintFromBytes(t[1][0], &result[0], LAMBDA_BYTES);
    std::cout << t[1][0] << std::endl;
    std::cout << tmp[1] << std::endl;

    // for(int j = 0; j < 2; j++){     
    //     // k is used for left and right
    //     bytesFromBigint(&seed[j][0], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][lambda_bytes], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][2*lambda_bytes], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][3*lambda_bytes], tmp[j], lambda_bytes);
        
    //     encryptwrapper(&seed[j][0], 2*(2*lambda_bytes+1), j);
    //     bigintFromBytes(t[0][j], &seed[j][0],1);
    //     t[0][j].get_mpz_t()->_mp_d[0] = t[0][j].get_mpz_t()->_mp_d[0]%2;
    //     bigintFromBytes(v[0][j], &seed[j][1],lambda_bytes);
    //     bigintFromBytes(s[0][j], &seed[j][lambda_bytes+1],lambda_bytes);

    //     bigintFromBytes(t[1][j], &seed[j][2*lambda_bytes+1],1);
    //     t[1][j].get_mpz_t()->_mp_d[0] = t[1][j].get_mpz_t()->_mp_d[0]%2;
    //     bigintFromBytes(v[1][j], &seed[j][2*lambda_bytes+2],lambda_bytes);
    //     bigintFromBytes(s[1][j], &seed[j][3*lambda_bytes+2],lambda_bytes);
        
    //     std::cout << "t is " << t[0][j] << std::endl;
    //     std::cout << "v is " << v[0][j] << std::endl;
    //     std::cout << "s is " << s[0][j] << std::endl;
        
    //     std::cout << "t is " << t[1][j] << std::endl;
    //     std::cout << "v is " << v[1][j] << std::endl;
    //     std::cout << "s is " << s[1][j] << std::endl;
        
    // }
 
    // std::cout << "-------------------" << std::endl;
    // tmp[0] = 1756036384109483092;
    // tmp[1] = 3326679033982098902;

    // for(int j = 0; j < 2; j++){     
    //     // k is used for left and right
    //     bytesFromBigint(&seed[j][0], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][lambda_bytes], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][2*lambda_bytes], tmp[j], lambda_bytes);
    //     bytesFromBigint(&seed[j][3*lambda_bytes], tmp[j], lambda_bytes);
        
    //     // encryptwrapper(&seed[j][0], 2*(2*lambda_bytes+1), j);
    //     bigintFromBytes(t[0][j], &seed[j][0],1);
    //     t[0][j].get_mpz_t()->_mp_d[0] = t[0][j].get_mpz_t()->_mp_d[0]%2;
    //     bigintFromBytes(v[0][j], &seed[j][1],lambda_bytes);
    //     bigintFromBytes(s[0][j], &seed[j][lambda_bytes+1],lambda_bytes);

    //     bigintFromBytes(t[1][j], &seed[j][2*lambda_bytes+1],1);
    //     t[1][j].get_mpz_t()->_mp_d[0] = t[1][j].get_mpz_t()->_mp_d[0]%2;
    //     bigintFromBytes(v[1][j], &seed[j][2*lambda_bytes+2],lambda_bytes);
    //     bigintFromBytes(s[1][j], &seed[j][3*lambda_bytes+2],lambda_bytes);
        
    //     std::cout << "t is " << t[0][j] << std::endl;
    //     std::cout << "v is " << v[0][j] << std::endl;
    //     std::cout << "s is " << s[0][j] << std::endl;
        
    //     std::cout << "t is " << t[1][j] << std::endl;
    //     std::cout << "v is " << v[1][j] << std::endl;
    //     std::cout << "s is " << s[1][j] << std::endl;
        
    // }
    // bytesFromBigint(&inp[0], x_add_r, LAMBDA_BYTES);
    // encryptwrapper(&inp[0], LAMBDA_BYTES, 2);
    // bigintFromBytes(result, &inp[0], LAMBDA_BYTES);
    // std::cout << "2nd result is " << result << std::endl;
    
    // bytesFromBigint(&inp[0], x_add_r, LAMBDA_BYTES);
    // encryptwrapper(&inp[0], LAMBDA_BYTES, 3);
    // bigintFromBytes(result, &inp[0], LAMBDA_BYTES);
    // std::cout << "3rd result is " << result << std::endl;
    // std::cout << "result is " << inp << std::endl;

    return 0;
}