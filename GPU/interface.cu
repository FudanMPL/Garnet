#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"

//aes加密
void encryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;
  uint8_t *w_d;
  uint8_t *w;

  cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t)*256);

  //为扩展后密钥分配内存空间
  w = (uint8_t*)malloc(240*sizeof(uint8_t));
  
  aes_key_expansion(key, w);

  //为数据和扩展后的密钥分配显存空间
  cudaMalloc((void**)&buf_d, numbytes);
  cudaMalloc((void**)&w_d, 240*sizeof(uint8_t));
  //从内存拷贝至显存
  cudaMemcpy(buf_d, buf, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(w_d, w, 240*sizeof(uint8_t), cudaMemcpyHostToDevice);

  //计算GRIDSIZE与BLOCKSIZE
  dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  //对每个数据块进行aes加密
  aes256_encrypt_ecb<<<dimBlock, dimGrid>>>(buf_d, numbytes, w_d);

  cudaMemcpy(buf, buf_d, numbytes, cudaMemcpyDeviceToHost);
  
}

// aes解密
void decryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;
  
  uint8_t *w;

  cudaMemcpyToSymbol(sboxinv, sboxinv, sizeof(uint8_t)*256);

  printf("\nBeginning decryption\n");

  //记录解密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);

  //为扩展后密钥分配内存空间
  w = (uint8_t*)malloc(240*sizeof(uint8_t));

  aes_key_expansion(key, w);

  //分配显存空间
  cudaMalloc((void**)&buf_d, numbytes);
  //从内存拷贝至显存
  cudaMemcpy(buf_d, buf, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(w3, w, 240*sizeof(uint8_t));

  //计算GRIDSIZE与BLOCKSIZE
  dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);
    //对每个数据块进行aes解密
  aes256_decrypt_ecb<<<dimBlock, dimGrid>>>(buf_d, numbytes);

  cudaMemcpy(buf, buf_d, numbytes, cudaMemcpyDeviceToHost);

  //记录解密算法结束时间，并计算解密速度
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("time:%f\n",total);
  printf("Throughput: %fGbps\n", numbytes/total/1024/1024/1024*8);
}

void test_add(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes){
  uint8_t *buf_a;
  uint8_t *buf_b;
  uint8_t *buf_res;

  cudaMalloc((void**)&buf_a, numbytes);
  cudaMalloc((void**)&buf_b, numbytes);
  cudaMalloc((void**)&buf_res, numbytes);
  //从内存拷贝至显存
  cudaMemcpy(buf_a, a, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(buf_b, b, numbytes, cudaMemcpyHostToDevice);

  _add<<<1, 1>>>(buf_a, buf_b, buf_res, numbytes);
  cudaMemcpy(res, buf_res, numbytes, cudaMemcpyDeviceToHost);
  return;
}

void test_restricted_multiply(int value, uint8_t * a, uint8_t * res, int numbytes){
  uint8_t *buf_a;
  uint8_t *buf_res;
  int *buf_value;

  cudaMalloc((void**)&buf_a, numbytes);
  cudaMalloc((void**)&buf_res, numbytes);
  cudaMalloc((void**)&buf_value, sizeof(int));
  //从内存拷贝至显存
  cudaMemcpy(buf_a, a, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(buf_value, &value, sizeof(int), cudaMemcpyHostToDevice);

  _restricted_multiply<<<1,numbytes>>>(buf_value, buf_a,  buf_res, numbytes);
  cudaMemcpy(res, buf_res, numbytes, cudaMemcpyDeviceToHost);
  return;
}

void test_xor(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes){
  uint8_t *buf_a;
  uint8_t *buf_b;
  uint8_t *buf_res;

  cudaMalloc((void**)&buf_a, numbytes);
  cudaMalloc((void**)&buf_b, numbytes);
  cudaMalloc((void**)&buf_res, numbytes);
  //从内存拷贝至显存
  cudaMemcpy(buf_a, a, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(buf_b, b, numbytes, cudaMemcpyHostToDevice);

  _xor<<<1, numbytes>>>(buf_a, buf_b, buf_res, numbytes);
  cudaMemcpy(res, buf_res, numbytes, cudaMemcpyDeviceToHost);
  return;
}

//uint8_t * seed_0, uint8_t * seed_1分别是初始化后的随机数种子
//uint8_t * generated_value_cpu_0是表示给party0生成的随机数结果存放位置， uint8_t * generated_value_cpu_1是表示给party1生成的随机数结果存放位置
void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * key, uint8_t * generated_value_cpu_0, uint8_t * generated_value_cpu_1, int numbytes){
    if(numbytes!=8 && numbytes!=16){
      printf("only support 64 or 128 bits");
      return;
    }
    uint8_t * generated_value_0;
    uint8_t * generated_value_1;
    uint8_t * w_d;
    uint8_t * w;
    
    cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t)*256);
   

    int bit_length = numbytes * 8;
    //分配扩展密钥
    w = (uint8_t*)malloc(240*sizeof(uint8_t));
    aes_key_expansion(key, w);

    cudaMalloc((void**)&w_d, 240*sizeof(uint8_t));

    // //分配输出的generated_value的长度
    cudaMalloc((void**)&generated_value_0, (bit_length - 1) * 2 * (numbytes + 1) + numbytes);
    cudaMalloc((void**)&generated_value_1, (bit_length - 1) * 2 * (numbytes + 1) + numbytes);
    //分配数据结构空间
    
     if(numbytes = 8){
      fss_gen_struct_64 * fss_gen = {nullptr};
      cudaMalloc((void**)&fss_gen, sizeof(fss_gen_struct_64));
      // cudaMalloc((void**)&fss_gen->va, numbytes);
      // cudaMalloc((void**)&(fss_gen->keep), sizeof(int));
      // cudaMalloc((void**)&(fss_gen->lose), sizeof(int));
      // for(int idx = 0 ; idx < 2 ; idx++){
      //     cudaMalloc((void**)&fss_gen->seed[idx], numbytes);
      //     cudaMalloc((void**)&fss_gen->scw[idx], numbytes);
      //     cudaMalloc((void**)&fss_gen->vcw[idx], numbytes);
      //     cudaMalloc((void**)&fss_gen->tcw[idx], 1);
      //     cudaMalloc((void**)&fss_gen->convert[idx], numbytes);
      //     cudaMalloc((void**)&fss_gen->convert_seed[idx], numbytes);
      //     cudaMalloc((void**)&fss_gen->inter_val[idx], 2*numbytes+1);
      //     for(int jdx = 0; jdx < 2; jdx++){
      //         cudaMalloc((void**)&fss_gen->s[idx][jdx], numbytes);
      //         cudaMalloc((void**)&fss_gen->v[idx][jdx], numbytes);
      //         cudaMalloc((void**)&fss_gen->t[idx][jdx], 1);
      //         cudaMalloc((void**)&fss_gen->pre_t[idx][jdx], 1);
      //     }
      // }
      // // //初始化种子
      // cudaMemcpy(w_d, w, 240*sizeof(uint8_t), cudaMemcpyHostToDevice);
      // cudaMemcpy(fss_gen->r, r, numbytes, cudaMemcpyHostToDevice);
      // cudaMemcpy(fss_gen->seed[0], seed0, numbytes, cudaMemcpyHostToDevice);
      // cudaMemcpy(fss_gen->seed[1], seed1, numbytes, cudaMemcpyHostToDevice);
      // cudaMemset(fss_gen->va, 0, cudaMemcpyHostToDevice);
      // cudaMemset(fss_gen->pre_t[0], 0, cudaMemcpyHostToDevice);
      // cudaMemset(fss_gen->pre_t[1], 1, cudaMemcpyHostToDevice);
        printf("space prepared!");
        dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
        dim3 dimGrid(THREADS_PER_BLOCK);

        fss_generate_gpu_64<<<dimBlock, dimGrid>>>(fss_gen, w_d, generated_value_cpu_0, generated_value_1, numbytes);
    }
    else{
      fss_gen_struct_128 * fss_gen = {nullptr};
      cudaMalloc((void**)&fss_gen, sizeof(fss_gen_struct_128));
      dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
      dim3 dimGrid(THREADS_PER_BLOCK);
      fss_generate_gpu_128<<<dimBlock, dimGrid>>>(fss_gen, w_d, generated_value_cpu_0, generated_value_1, numbytes);
    }


}

