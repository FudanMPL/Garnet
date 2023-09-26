/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:02:44
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-09-15 10:16:17
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
    
    
    bigint bseed, t, v, s;
    octet seed[LAMBDA_BYTES], result[LAMBDA_BYTES];
    // generate key
    
    //读入数据
    k_in.open("./build/k1", std::ios::in);
    k_in >> bseed;
    

    bytesFromBigint(&seed[0], bseed, LAMBDA_BYTES);
    encryptwrapper(&seed[0], 1, 1);
    bigintFromBytes(t, &seed[0],1);

    std::cout << "t is " << t << std::endl;
    std::cout << "t%2 is " << (t.get_mpz_t()->_mp_d[0]%2) << std::endl;


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