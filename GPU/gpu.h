/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:06:03
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-03 23:36:52
 * @FilePath: /txy/Garnet/GPU/gpu.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "Math/bigint.h"
#include <iostream>

void encryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes);

void add(uint8_t *a, uint8_t *b, uint8_t *res, int length);

void restricted_multiply(int value, uint8_t * a, uint8_t *res, int length);
 
// void decryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes);

void encryptwrapper(uint8_t *buf, unsigned long numbytes, int time){
    //调用次数最多是2^31-1次
    uint8_t key[32];
    int len = sizeof(key);
    for (int i = 0; i < len; i++){
        key[i] = i*(time+1);
    }
    encryptdemo(key, buf, numbytes);
}

void test_add(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void test_restricted_multiply(int value, uint8_t * a, uint8_t * res, int numbytes);

void test_xor(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * key, uint8_t * generated_value_cpu_0, uint8_t * generated_value_cpu_1, int numbytes);