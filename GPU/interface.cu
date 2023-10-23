#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"


//uint8_t * seed_0, uint8_t * seed_1分别是初始化后的随机数种子
//uint8_t * generated_value_cpu_0是表示给party0生成的随机数结果存放位置， uint8_t * generated_value_cpu_1是表示给party1生成的随机数结果存放位置
void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel){
  uint8_t * generated_value;

  // 判断fss_struct.h中预设的BYTE_LEN即输入的环的大小是否与当前计算的相等
  if(numbytes != BYTE_LEN){
    printf("numbytes is %d, BYTE_LEN is %d, numbytes!=BYTE_LEN, please configure the BYTE_LEN!", numbytes, BYTE_LEN);
  }

  // preset value
  int lambda = 127;
  int block_number = parallel;
  int bit_length = numbytes * 8;
  
  //分配扩展密钥
  BYTE key[240];
  int keyLen = 16;
  int blockLen = 16;

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount; 
  BYTE *cuda_key[2];//, *cuda_Sbox;
  int expandKeyLen = AES_ExpandKey(key, keyLen);
  // std::cout << "expandKeyLen is " << expandKeyLen << std::endl;
  int thrdperblock = block_number/num_sm;
  for(int i = 0; i < keyLen; i++){
      key[i] = i;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key[0], 240);
  cudaMemcpy(cuda_key[0], key, 240, cudaMemcpyHostToDevice); 
  
  
  for (int i = 0; i < keyLen; i++){
      key[i] = key[i] * 2;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key[1], 240);
  cudaMemcpy(cuda_key[1], key, 240, cudaMemcpyHostToDevice); 

  if(block_number%num_sm>0)
      thrdperblock++;

  if(thrdperblock>1024){
      thrdperblock = 1024;
      num_sm = block_number/1024;
      if(block_number%1024>0){
          num_sm++;
      }
  }
  dim3 ThreadperBlock(thrdperblock);
  dim3 BlockperGrid(num_sm);

  // //分配输出的generated_value的长度
  cudaMalloc((void**)&generated_value, ((bit_length) * 2 * BYTE_LAMBDA + BYTE_LEN) * sizeof(BYTE));
  //分配数据结构空间

  FssGen * fss_gen = new FssGen();
  FssGen * cuda_fss_gen;
  for(int i = 0; i < 2; i++){
    fss_gen->pre_t[i] = i;
  }
  
  cudaMalloc(&cuda_fss_gen, sizeof(class FssGen));
  cudaMemcpy(cuda_fss_gen->pre_t, fss_gen->pre_t, 2*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_fss_gen->seed[0], seed0, BYTE_LAMBDA * sizeof(BYTE), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_fss_gen->seed[1], seed1, BYTE_LAMBDA * sizeof(BYTE), cudaMemcpyHostToDevice);
  
  //记录加密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);


  int keep, lose;
  for(int i = 0; i < bit_length; i++){
    int idx = int(i/8);
    keep = ((r[idx]) >> (7 - (i - (idx) * 8)))%2;
    lose = keep ^ 1;       
    // keep 和 lose 正确
    // std::cout << "keep is " << keep << " lose is " << lose << std::endl;
    for(int j = 0; j < 2; j++){
      for(int k = 0; k < 2; k++){
        rand_svt_generate<<<1,1>>>(cuda_fss_gen, cuda_key[k], j, k, num_sm, thrdperblock);
      }
    }
    _xor<<<1,BYTE_LAMBDA>>>(cuda_fss_gen->s[lose][0], cuda_fss_gen->s[lose][1], cuda_fss_gen->scw, BYTE_LAMBDA);
   
    for(int j = 0; j < 2; j++){
      _copy<<<1,BYTE_LEN>>>(cuda_fss_gen->v[lose][j], cuda_fss_gen->convert[j], 0, 0, BYTE_LEN);
    }
    vcw_generate_update_pre_t<<<1,1>>>(cuda_fss_gen, keep);
    if(keep){
      vcw_generate_update_keep<<<1,1>>>(cuda_fss_gen);
    }
    for(int j = 0; j < 2; j++){
        _copy<<<1,BYTE_LEN>>>(cuda_fss_gen->v[keep][j], cuda_fss_gen->convert[j], 0, 0, BYTE_LEN);
    }
    va_genearte_update<<<1,1>>>(cuda_fss_gen, keep);
    stcw_pret_generate_update<<<1,1>>>(cuda_fss_gen, keep);
    generated_value_update<<<1,1>>>(cuda_fss_gen, generated_value, i);
    
  }

  for(int j = 0; j < 2; j++){
    _copy<<<1,BYTE_LEN>>>(cuda_fss_gen->seed[j], cuda_fss_gen->convert[j], 0, 0, BYTE_LEN);
  }
  
  final_cw_generate_update<<<1,1>>>(cuda_fss_gen, generated_value, bit_length, BYTE_LEN);

  //记录加密算法结束时间，并计算加密速度
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("time:%f\n",total);


  cudaMemcpy(generated_value_cpu, generated_value, ((bit_length) * 2 * BYTE_LAMBDA + BYTE_LEN) * sizeof(BYTE), cudaMemcpyDeviceToHost);

  cudaFree(cuda_fss_gen);
  cudaFree(cuda_key);
  cudaFree(generated_value);
}


void fss_evaluate(int party, uint8_t * x_reveal, uint8_t * seed, uint8_t * generated_value_cpu, uint8_t * result, int numbytes, int parallel){
  int bit_length = numbytes * 8, lambda_bytes = 16, lambda = 127;
  // 判断fss_struct.h中预设的BYTE_LEN即输入的环的大小是否与当前计算的相等
  if(numbytes != BYTE_LEN){
    printf("numbytes is %d, BYTE_LEN is %d, numbytes!=BYTE_LEN, please configure the BYTE_LEN!", numbytes, BYTE_LEN);
  }
  uint8_t * generated_value;
  uint8_t * cuda_result;
  //分配扩展密钥
  BYTE key[240];
  int keyLen = 16;
  int blockLen = 16;

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount; 
  BYTE *cuda_key[2];//, *cuda_Sbox;
  int expandKeyLen = AES_ExpandKey(key, keyLen);
  std::cout << "expandKeyLen is " << expandKeyLen << std::endl;
  int thrdperblock = parallel/num_sm;

  // 和seed无关
  for(int i = 0; i < keyLen; i++){
      key[i] = i;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key[0],240);
  cudaMemcpy(cuda_key[0], key, 240, cudaMemcpyHostToDevice); 

  for (int i = 0; i < keyLen; i++){
      key[i] = key[i] * 2;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key[1], 240);
  cudaMemcpy(cuda_key[1], key, 240, cudaMemcpyHostToDevice); 
  if(parallel%num_sm>0)
      thrdperblock++;

  if(thrdperblock>1024){
      thrdperblock = 1024;
      num_sm = parallel/1024;
      if(parallel%1024>0){
          num_sm++;
      }
  }
  dim3 ThreadperBlock(thrdperblock);
  dim3 BlockperGrid(num_sm);
  
  cudaMalloc((void**)&generated_value, (((bit_length) * 2 * BYTE_LAMBDA + BYTE_LEN) * sizeof(BYTE)));
  //分配数据结构空间
  FssEval * fss_eval = new FssEval();
  FssEval * cuda_fss_eval;
  fss_eval->pre_t = party;
  

  // for(int i = 0; i < 2; i++)
  //   printGpuBytes<<<1,1>>>(cuda_key[i], 240);

  cudaMalloc(&cuda_result, numbytes);
  cudaMalloc(&cuda_fss_eval, sizeof(class FssEval));
  cudaMemcpy(generated_value, generated_value_cpu, (((bit_length) * 2 * BYTE_LAMBDA + BYTE_LEN)) * sizeof(BYTE), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_fss_eval, fss_eval, sizeof(class FssEval), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_fss_eval->seed, seed, numbytes * sizeof(BYTE), cudaMemcpyHostToDevice);
  //记录加密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);
  for(int i = 0; i < bit_length; i++){
    int idx = int(i/8);
    int xi = ((x_reveal[idx]) >> (7 - (i - (idx) * 8)))%2;
    cw_eval_copy<<<1,1>>>(cuda_fss_eval, generated_value, i);
    for(int j = 0; j < 2; j++){
      rand_svt_eval_generate<<<1,1>>>(cuda_fss_eval, cuda_key[j], j, num_sm, thrdperblock);
    }
    st_eval_phrase<<<1,2>>>(cuda_fss_eval);    
    value_eval_update<<<1,1>>>(cuda_fss_eval, cuda_key[0], xi, party, num_sm, thrdperblock);    
  }
  final_eval_update<<<1,1>>>(cuda_fss_eval, cuda_key[0], generated_value, cuda_result, party, num_sm, thrdperblock);
  
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("eval time:%f\n",total);
  
  cudaMemcpy(result, cuda_result,  numbytes, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++)
    std::cout << result[i];
  printf("\n");
  cudaFree(cuda_fss_eval);
  cudaFree(cuda_key);
  cudaFree(generated_value);
}
