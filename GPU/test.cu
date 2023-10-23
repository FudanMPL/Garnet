#include "utils.cu"

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
