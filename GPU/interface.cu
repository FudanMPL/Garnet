#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"

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

  cudaMalloc((void**)&buf_a, numbytes);
  cudaMalloc((void**)&buf_res, numbytes);
  //从内存拷贝至显存
  cudaMemcpy(buf_a, a, numbytes, cudaMemcpyHostToDevice);

  _restricted_multiply<<<1,numbytes>>>(value, buf_a,  buf_res, numbytes);
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

void test_sub(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes){
  uint8_t *buf_a;
  uint8_t *buf_b;
  uint8_t *buf_res;

  cudaMalloc((void**)&buf_a, numbytes);
  cudaMalloc((void**)&buf_b, numbytes);
  cudaMalloc((void**)&buf_res, numbytes);
  //从内存拷贝至显存
  cudaMemcpy(buf_a, a, numbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(buf_b, b, numbytes, cudaMemcpyHostToDevice);

  _sub<<<1, 1>>>(buf_a, buf_b, buf_res, numbytes);
  cudaMemcpy(res, buf_res, numbytes, cudaMemcpyDeviceToHost);
  return;
}

//uint8_t * seed_0, uint8_t * seed_1分别是初始化后的随机数种子
//uint8_t * generated_value_cpu_0是表示给party0生成的随机数结果存放位置， uint8_t * generated_value_cpu_1是表示给party1生成的随机数结果存放位置
void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel){
  uint8_t * generated_value;
  
  int block_number = parallel;
  int bit_length = numbytes * 8;
  
  //分配扩展密钥
  BYTE key[16 * (14 + 1)];
  int keyLen = 16;
  int blockLen = 16;

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount; 
  BYTE *cuda_key;//, *cuda_Sbox;
  int expandKeyLen = AES_ExpandKey(key, keyLen);
  int thrdperblock = block_number/num_sm;
  for(int i = 0; i < blockLen; i++){
      key[i] = i;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key,16*15*sizeof(BYTE) );
  cudaMemcpy(cuda_key, key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice); 
  
  
  // for (int i = 0; i < blockLen; i++){
  //     key[i] = key[i] * 2;
  // }
  // AES_ExpandKey(key, keyLen);
  // cudaMalloc(&cuda_key[1],16*15*sizeof(BYTE) );
  // cudaMemcpy(cuda_key[1], key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice); 

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
  cudaMalloc((void**)&generated_value, (bit_length - 1) * 2 * (numbytes + 1) + numbytes);
  //分配数据结构空间
  
  FssGen * fss_gen = new FssGen();
  FssGen * cuda_fss_gen;
  for(int i = 0; i < 2; i++){
    fss_gen->pre_t[i] = i;
  }
  
  cudaMalloc(&cuda_fss_gen, sizeof(class FssGen));
  cudaMemcpy(&cuda_fss_gen->pre_t, fss_gen->pre_t, 2*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&cuda_fss_gen->seed[0], seed0, numbytes * sizeof(BYTE), cudaMemcpyHostToDevice);
  cudaMemcpy(&cuda_fss_gen->seed[1], seed1, numbytes * sizeof(BYTE), cudaMemcpyHostToDevice);
  
  //记录加密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);

  int keep, lose;
  for(int i = 0; i < bit_length-1; i++){
    int idx = int(i/8);
    keep = ((r[idx]) >> (7 - (i - (idx) * 8)))%2;
    lose = keep ^ 1;       
    
    for(int j = 0; j < 2; j++){
      _copy<<<1,numbytes>>>(cuda_fss_gen->seed[j], cuda_fss_gen->inter_val[j], 0, 0, numbytes);
      _copy<<<1,numbytes>>>(cuda_fss_gen->seed[j], cuda_fss_gen->inter_val[j], 0, numbytes, numbytes);
      _copy<<<1,numbytes>>>(cuda_fss_gen->seed[j], cuda_fss_gen->inter_val[j], 0, 2*numbytes, 1);
      
      // printGpuBytes<<<1,1>>>(cuda_fss_gen->inter_val[j], 2*numbytes+1);
      
      for(int k = 0; k < 2; k++){
        AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->inter_val[j], cuda_key, 176, numbytes, 3);
        _copy<<<1,numbytes>>>(cuda_fss_gen-> inter_val[j], cuda_fss_gen->t[j][0], 0, 0, 1);
        _mod2_t<<<1,1>>>(cuda_fss_gen->t[j][0]);
        _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->v[j][0], 1, 0, numbytes);
        _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->s[j][0], numbytes + 1, 0,  numbytes);

        AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->inter_val[j], cuda_key, 176, numbytes, 3);
        _copy<<<1,numbytes>>>(cuda_fss_gen-> inter_val[j], cuda_fss_gen->t[j][1],  0, 0, 1);
        _mod2_t<<<1,1>>>(cuda_fss_gen->t[j][1]);
        _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->v[j][1], 1, 0, numbytes);
        _copy<<<1,numbytes>>>(cuda_fss_gen->inter_val[j], cuda_fss_gen->s[j][1], 1 + numbytes, 0, numbytes);
      }
    }
    _xor<<<1,numbytes>>>(cuda_fss_gen->s[lose][0], cuda_fss_gen->s[lose][1], cuda_fss_gen->scw, numbytes);
    for(int j = 0; j < 2; j++){
        _copy<<<1,numbytes>>>(cuda_fss_gen->v[lose][j], cuda_fss_gen->convert_seed[j], 0, 0, numbytes);
        AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->seed[j], cuda_key, 176, numbytes, 1);
        _copy<<<1,numbytes>>>(cuda_fss_gen->convert_seed[j], cuda_fss_gen->convert[i], 0, 0, numbytes);
    }
    if(keep){
      vcw_generate_update_keep<<<1,1>>>(cuda_fss_gen, numbytes);
    }
    else{
      vcw_generate_update_lose<<<1,1>>>(cuda_fss_gen, numbytes);
    }
    for(int j = 0; j < 2; j++){
        _copy<<<1,numbytes>>>(cuda_fss_gen->v[keep][j], cuda_fss_gen->convert_seed[j], 0, 0, numbytes);
        AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->convert_seed[j], cuda_key, 176, numbytes, 1);
        _copy<<<1,numbytes>>>(cuda_fss_gen->convert_seed[j], cuda_fss_gen->convert[j], 0, 0, numbytes);
    }
    va_genearte_update<<<1,1>>>(cuda_fss_gen, keep, numbytes);
    tcw_pret_generate_update<<<1,1>>>(cuda_fss_gen, keep, numbytes);
    _copy<<<1,numbytes>>>(cuda_fss_gen->scw, generated_value, 0, 2*(numbytes+1)*i, numbytes);
    _copy<<<1,numbytes>>>(cuda_fss_gen->vcw, generated_value, 0, 2*(numbytes+1)*i + numbytes, numbytes); 
    _copy<<<1,numbytes>>>(cuda_fss_gen->tcw[0], generated_value, 0, 2*(numbytes+1)*i + 2 * numbytes, 1);
    _copy<<<1,numbytes>>>(cuda_fss_gen->tcw[1], generated_value, 0, 2*(numbytes+1)*i + 2 * numbytes + 1, 1);
  }
  
  for(int j = 0; j < 2; j++){
      _copy<<<1,numbytes>>>(cuda_fss_gen->seed[j], cuda_fss_gen->convert_seed[j], 0, 0, numbytes);
      AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_fss_gen->convert_seed[j], cuda_key, 176, numbytes, 1);
      _copy<<<1,numbytes>>>(cuda_fss_gen->convert_seed[j], cuda_fss_gen->convert[j], 0, 0, numbytes);
  }
  
  final_cw_generate_update<<<1,1>>>(cuda_fss_gen, generated_value, bit_length, numbytes);

  //记录加密算法结束时间，并计算加密速度
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("time:%f\n",total);


  cudaMemcpy(generated_value_cpu, generated_value, (bit_length - 1) * 2 * (numbytes + 1) + numbytes, cudaMemcpyDeviceToHost);

}


void fss_evaluate(int party, uint8_t * x_reveal, uint8_t * seed, uint8_t * gen_val, uint8_t * result, int numbytes, int parallel){
  int bit_length = numbytes * 8;
  uint8_t * generated_value;
  //分配扩展密钥
  BYTE key[16 * (14 + 1)];
  int keyLen = 16;
  int blockLen = 16;

  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount; 
  BYTE *cuda_key;//, *cuda_Sbox;
  int expandKeyLen = AES_ExpandKey(key, keyLen);
  int thrdperblock = parallel/num_sm;
  for(int i = 0; i < blockLen; i++){
      key[i] = i;
  }
  AES_ExpandKey(key, keyLen);
  cudaMalloc(&cuda_key,16*15*sizeof(BYTE) );
  cudaMemcpy(cuda_key, key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice); 
  
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
  
  // //分配输出的generated_value的长度
  cudaMalloc((void**)&generated_value, (bit_length - 1) * 2 * (numbytes + 1) + numbytes);
  //分配数据结构空间
  
  FssEval * fss_eval = new FssEval();
  FssEval * cuda_fss_eval;
  fss_eval->pre_t = party;
  
  cudaMalloc(&cuda_fss_eval, sizeof(class FssEval));
  cudaMemcpy(&cuda_fss_eval, fss_eval, sizeof(class FssEval), cudaMemcpyHostToDevice);
  cudaMemcpy(&cuda_fss_eval->seed, seed, numbytes * sizeof(BYTE), cudaMemcpyHostToDevice);

  //记录加密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);
  for(int i = 0; i < bit_length-1; i++){
    int idx = int(i/8);
    int xi = ((x_reveal[idx]) >> (7 - (i - (idx) * 8)))%2;
    correction_word_eval_copy<<<1,1>>>(cuda_fss_eval, generated_value, numbytes, i);
    random_value_eval_generate<<<1,2>>>(cuda_fss_eval, cuda_key, numbytes, num_sm, thrdperblock);
    convert_random_value_eval_generate<<<1,2>>>(cuda_fss_eval, cuda_key, numbytes, num_sm, thrdperblock);
    value_eval_update<<<1,1>>>(cuda_fss_eval, xi, party, numbytes);
  }
  final_eval_update<<<1,1>>>(cuda_fss_eval, cuda_key, generated_value, party, numbytes, num_sm, thrdperblock);
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("eval time:%f\n",total);
  cudaMemcpy(cuda_fss_eval->tmp_v, result, numbytes, cudaMemcpyDeviceToHost);
}