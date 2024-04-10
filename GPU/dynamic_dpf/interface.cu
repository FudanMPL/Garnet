#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"
#include "test.cu"
#include <thrust/device_vector.h>
#include <cmath>



void fss_dpf_generate(InputByteRelatedValuesGen cpu_values, aes_gen_block * cpu_aes_block_array, int input_length, int parallel){ 
    BYTE key[240];
    int keyLen = 16;
    int input_byte = ceil(input_length / 8);
    KeyBlock * cuda_key_block;
    gpuErrorCheck(cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock)));
    prepare_key(cuda_key_block, key, keyLen);

    // printGpuBytes<<<1,1>>>(cuda_key_block->cuda_key[0], 0, 240);
    int thrdperblock, num_sm, rounds;
    init_sm_thrd(num_sm, thrdperblock, parallel, rounds);
    // std::cout << num_sm << " " << thrdperblock << " " << rounds << std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);


    FssDpfGen * cuda_dpf_gen;   
    aes_gen_block * cuda_aes_block_array;
    
    // input length related values
    // random values, shape = [parallel, input_byte]
    uint8_t * cuda_r;
    gpuErrorCheck(cudaMalloc(&cuda_r, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemcpy(cuda_r, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // correction words, scw.shape = [parallel, input_length, input_byte]
    uint8_t * cuda_scw;
    gpuErrorCheck(cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t)));
    // tcw.shape = [parallel, input_length]
    bool * cuda_tcw_0;
    bool * cuda_tcw_1;
    gpuErrorCheck(cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool)));
    gpuErrorCheck(cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool)));
    // output.shape = [parallel, input_byte]
    uint8_t * cuda_output;
    gpuErrorCheck(cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen)));
    gpuErrorCheck(cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_gen_block)));
    gpuErrorCheck(cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_gen_block), cudaMemcpyHostToDevice));
    
    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    for(int round = 0; round <= rounds; round++){
        dpf_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, cuda_dpf_gen, cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_output, input_length, input_byte, parallel, round, num_sm, thrdperblock);
    }

    gpuErrorCheck(cudaMemcpy(cpu_values.scw, cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.tcw[0], cuda_tcw_0, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.tcw[1], cuda_tcw_1, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.output, cuda_output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    // printf("eval time:%f\n",total);
    gpuErrorCheck(cudaFree(cuda_dpf_gen));
    gpuErrorCheck(cudaFree(cuda_aes_block_array));
    gpuErrorCheck(cudaFree(cuda_r));
    gpuErrorCheck(cudaFree(cuda_output));
    gpuErrorCheck(cudaFree(cuda_scw));
    gpuErrorCheck(cudaFree(cuda_tcw_0));
    gpuErrorCheck(cudaFree(cuda_tcw_1));
}
// // 原版的fss_dpf_evaluate里面将Gen的参数也传入进来了，但实际上只需要将Eval的参数传入即可。将拷贝放在CPU上进行。
void fss_dpf_evaluate(InputByteRelatedValuesEval cpu_eval_values, InputByteRelatedValuesGen cpu_values, aes_eval_block * cpu_aes_block_array, bool party, int input_length, int parallel){
    int input_byte = ceil(input_length/8);  
    BYTE key[240];
    int keyLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
    prepare_key(cuda_key_block, key, keyLen); 

    int thrdperblock, num_sm, rounds;
    init_sm_thrd(num_sm, thrdperblock, parallel, rounds);
    // std::cout << num_sm << " " << thrdperblock <<  std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);

    
    FssDpfEval * cuda_dpf_eval;   
    aes_eval_block * cuda_aes_block_array;
    uint8_t * cuda_reveal;
    uint8_t * cuda_scw;
    bool * cuda_tcw_0;
    bool * cuda_tcw_1;
    uint8_t * cuda_output;
    uint8_t * cuda_result;

    // input length related values
    // random values, shape = [parallel, input_byte]
    
    gpuErrorCheck(cudaMalloc(&cuda_reveal, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemcpy(cuda_reveal, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // correction words, scw.shape = [parallel, input_length, LAMBDA_BYTE]
    gpuErrorCheck(cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemcpy(cuda_scw, cpu_values.scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // tcw.shape = [parallel, input_length]
    gpuErrorCheck(cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool)));
    gpuErrorCheck(cudaMemcpy(cuda_tcw_0, cpu_values.tcw[0], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool)));
    gpuErrorCheck(cudaMemcpy(cuda_tcw_1, cpu_values.tcw[1], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice));
    // output.shape = [parallel, input_byte]
    gpuErrorCheck(cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemcpy(cuda_output, cpu_values.output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMalloc(&cuda_result, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMalloc(&cuda_dpf_eval, parallel*sizeof(class FssDpfEval)));
    gpuErrorCheck(cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_eval_block)));
    gpuErrorCheck(cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_eval_block), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    for(int round = 0; round <= rounds; round++){
        dpf_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, cuda_dpf_eval, cuda_reveal, cuda_scw, cuda_result, cuda_tcw_0, cuda_tcw_1, cuda_output, input_length, input_byte, party, parallel, round, num_sm, thrdperblock);
    }
    gpuErrorCheck(cudaMemcpy(cpu_eval_values.result, cuda_result, parallel*input_byte*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    gpuErrorCheck(cudaFree(cuda_dpf_eval));
    gpuErrorCheck(cudaFree(cuda_key_block));
    gpuErrorCheck(cudaFree(cuda_reveal));
    gpuErrorCheck(cudaFree(cuda_result));
    gpuErrorCheck(cudaFree(cuda_scw));
    gpuErrorCheck(cudaFree(cuda_tcw_0));
    gpuErrorCheck(cudaFree(cuda_tcw_1));
    gpuErrorCheck(cudaFree(cuda_output));
    gpuErrorCheck(cudaFree(cuda_aes_block_array));
}

// void fss_dpf_evaluate(InputByteRelatedValuesEval cpu_eval_values, aes_eval_block * cpu_aes_block_array, bool party, int input_length, int parallel){
//     int input_byte = ceil(input_length/8);  
//     BYTE key[240];
//     int keyLen = 16;
//     KeyBlock * cuda_key_block;
//     cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
//     prepare_key(cuda_key_block, key, keyLen); 

//     int thrdperblock, num_sm, rounds;
//     init_sm_thrd(num_sm, thrdperblock, parallel, rounds);
//     // std::cout << num_sm << " " << thrdperblock <<  std::endl;

//     dim3 ThreadperBlock(thrdperblock);
//     dim3 BlockperGrid(num_sm);

    
//     FssDpfEval * cuda_dpf_eval;   
//     aes_eval_block * cuda_aes_block_array;
//     uint8_t * cuda_reveal;
//     uint8_t * cuda_scw;
//     bool * cuda_tcw_0;
//     bool * cuda_tcw_1;
//     uint8_t * cuda_output;
//     uint8_t * cuda_result;

//     // input length related values
//     // random values, shape = [parallel, input_byte]
    
//     gpuErrorCheck(cudaMalloc(&cuda_reveal, parallel * input_byte * sizeof(uint8_t)));
//     gpuErrorCheck(cudaMemcpy(cuda_reveal, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
//     // correction words, scw.shape = [parallel, input_length, LAMBDA_BYTE]
//     gpuErrorCheck(cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t)));
//     gpuErrorCheck(cudaMemcpy(cuda_scw, cpu_values.scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyHostToDevice));
//     // tcw.shape = [parallel, input_length]
//     gpuErrorCheck(cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool)));
//     gpuErrorCheck(cudaMemcpy(cuda_tcw_0, cpu_values.tcw[0], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice));
//     gpuErrorCheck(cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool)));
//     gpuErrorCheck(cudaMemcpy(cuda_tcw_1, cpu_values.tcw[1], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice));
//     // output.shape = [parallel, input_byte]
//     gpuErrorCheck(cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t)));
//     gpuErrorCheck(cudaMemcpy(cuda_output, cpu_values.output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
//     gpuErrorCheck(cudaMalloc(&cuda_result, parallel * input_byte * sizeof(uint8_t)));
//     gpuErrorCheck(cudaMalloc(&cuda_dpf_eval, parallel*sizeof(class FssDpfEval)));
//     gpuErrorCheck(cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_eval_block)));
//     gpuErrorCheck(cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_eval_block), cudaMemcpyHostToDevice));

//     cudaDeviceSynchronize();
//     cudaEvent_t start1;
//     cudaEventCreate(&start1);
//     cudaEvent_t stop1;
//     cudaEventCreate(&stop1);
//     cudaEventRecord(start1);
//     for(int round = 0; round <= rounds; round++){
//         dpf_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, cuda_dpf_eval, cuda_reveal, cuda_scw, cuda_result, cuda_tcw_0, cuda_tcw_1, cuda_output, input_length, input_byte, party, parallel, round, num_sm, thrdperblock);
//     }
//     gpuErrorCheck(cudaMemcpy(cpu_eval_values.result, cuda_result, parallel*input_byte*sizeof(uint8_t), cudaMemcpyDeviceToHost));
//     cudaEventRecord(stop1);
//     cudaEventSynchronize(stop1);
//     float msecTotal1,total;
//     cudaEventElapsedTime(&msecTotal1, start1, stop1);
//     total=msecTotal1/1000;
//     printf("eval time:%f\n",total);
//     gpuErrorCheck(cudaFree(cuda_dpf_eval));
//     gpuErrorCheck(cudaFree(cuda_key_block));
//     gpuErrorCheck(cudaFree(cuda_reveal));
//     gpuErrorCheck(cudaFree(cuda_result));
//     gpuErrorCheck(cudaFree(cuda_scw));
//     gpuErrorCheck(cudaFree(cuda_tcw_0));
//     gpuErrorCheck(cudaFree(cuda_tcw_1));
//     gpuErrorCheck(cudaFree(cuda_output));
//     gpuErrorCheck(cudaFree(cuda_aes_block_array));
// }

void fss_dpf_generate_traverse(InputByteRelatedValuesGen cpu_values, aes_gen_block * cpu_aes_block_array, int input_length, int compress, int parallel){
    int input_byte = ceil(input_length/8);  
    input_length = input_length - compress;
    BYTE key[240];
    int keyLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
    prepare_key(cuda_key_block, key, keyLen);

    // printGpuBytes<<<1,1>>>(cuda_key_block->cuda_key[0], 0, 240);
    int thrdperblock, num_sm, rounds;
    init_sm_thrd(num_sm, thrdperblock, parallel, rounds);
    // std::cout << num_sm << " " << thrdperblock <<  std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);


    FssDpfGen * cuda_dpf_gen;   
    aes_gen_block * cuda_aes_block_array;
    
    // input length related values
    // random values, shape = [parallel, input_byte]
    uint8_t * cuda_r;
    gpuErrorCheck(cudaMalloc(&cuda_r, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemcpy(cuda_r, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // correction words, scw.shape = [parallel, input_length, input_byte]
    uint8_t * cuda_scw;
    gpuErrorCheck(cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t)));
    // tcw.shape = [parallel, input_length]
    bool * cuda_tcw_0;
    bool * cuda_tcw_1;
    gpuErrorCheck(cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool)));
    gpuErrorCheck(cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool)));
    // output.shape = [parallel, input_byte]
    uint8_t * cuda_output;
    gpuErrorCheck(cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t)));
    gpuErrorCheck(cudaMemset(cuda_output, 0, parallel * input_byte * sizeof(uint8_t)));
    // beta.shape = [parallel, input_byte]

    gpuErrorCheck(cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen)));
    gpuErrorCheck(cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_gen_block)));
    gpuErrorCheck(cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_gen_block), cudaMemcpyHostToDevice));
    
    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    for(int round = 0; round <= rounds; round++){
        dpf_compress_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, cuda_dpf_gen, cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_output, input_length, input_byte, compress, parallel, round, num_sm, thrdperblock);
    }
    
    // printGpuBytes<<<1,1>>>(cuda_output, 0, input_byte);
    gpuErrorCheck(cudaMemcpy(cpu_values.scw, cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.tcw[0], cuda_tcw_0, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.tcw[1], cuda_tcw_1, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(cpu_values.output, cuda_output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    // printf("eval time:%f\n",total);
    gpuErrorCheck(cudaFree(cuda_dpf_gen));
    gpuErrorCheck(cudaFree(cuda_aes_block_array));
    gpuErrorCheck(cudaFree(cuda_r));
    gpuErrorCheck(cudaFree(cuda_output));
    gpuErrorCheck(cudaFree(cuda_scw));
    gpuErrorCheck(cudaFree(cuda_tcw_0));
    gpuErrorCheck(cudaFree(cuda_tcw_1));
}


