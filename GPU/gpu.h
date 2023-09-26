/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-09-06 19:06:03
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-09-15 09:53:40
 * @FilePath: /txy/Garnet/GPU/gpu.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "Math/bigint.h"
#include <iostream>

void encryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes);

// void decryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes);

void encryptwrapper( uint8_t *buf, unsigned long numbytes, int time){
    //调用次数最多是2^31-1次
    uint8_t key[32];
    int len = sizeof(key);
    for (int i = 0; i < len; i++){
        key[i] = time%256;
        time = time >> 8;
    }
    encryptdemo(key, buf, numbytes);
}

// void decryptwrapper(uint8_t *key, uint8_t *buf, unsigned long numbytes){
//     encryptdemo(key, buf, numbytes);
// }

// void fsseval(uint8_t *key, uint8_t *buf, unsigned long numbytes);

// void fssevalwrapper(bigint init_key, bigint* scw, bigint * vcw, bigint * tcw_0, bigint * tcw_1, unsigned long numbytes){
//     std::cout << "calling fss eval wrapper!" << std::endl;
// }