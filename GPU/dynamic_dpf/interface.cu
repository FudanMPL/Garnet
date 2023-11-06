#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"
#include "test.cu"
#include <thrust/device_vector.h>
#include <cmath>

void fss_dpf_generate(InputByteRelatedValuesGen cpu_values, aes_gen_block * cpu_aes_block_array, int input_length, int parallel){
    int lambda = 127;  
    int input_byte = ceil(input_length/8);  
    BYTE key[240];
    int keyLen = 16;
    int blockLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
    prepare_key(cuda_key_block, key, keyLen);

    // printGpuBytes<<<1,1>>>(cuda_key_block->cuda_key[0], 0, 240);
    int thrdperblock, num_sm;
    init_sm_thrd(num_sm, thrdperblock, parallel);
    std::cout << num_sm << " " << thrdperblock <<  std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);


    FssDpfGen * cuda_dpf_gen;   
    aes_gen_block * cuda_aes_block_array;
    
    // input length related values
    // random values, shape = [parallel, input_byte]
    uint8_t * cuda_r;
    cudaMalloc(&cuda_r, parallel * input_byte * sizeof(uint8_t));
    cudaMemcpy(cuda_r, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // correction words, scw.shape = [parallel, input_length, input_byte]
    uint8_t * cuda_scw;
    cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t));
    // tcw.shape = [parallel, input_length]
    bool * cuda_tcw_0;
    bool * cuda_tcw_1;
    cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool));
    cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool));
    // output.shape = [parallel, input_byte]
    uint8_t * cuda_output;
    cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t));
    cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen));
    cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_gen_block));
    cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_gen_block), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    gen_init<<<BlockperGrid, ThreadperBlock>>>(cuda_dpf_gen, parallel);
    for(int i = 0; i < input_length; i++){        
        for(int j = 0; j < 2; j++){
            AES_Encrypt_Gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, 176, j, parallel);
            st_copy_gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_dpf_gen, j, parallel); 
        }       
        cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, input_byte, input_length, parallel);        
        for(int b = 0; b < 2; b++){
            st_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, b, input_byte, input_length, parallel);
        }
    }
    final_cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_output, cuda_dpf_gen, input_byte, parallel);
    cudaMemcpy(cpu_values.scw, cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.tcw[0], cuda_tcw_0, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.tcw[1], cuda_tcw_1, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.output, cuda_output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_gen);
    cudaFree(cuda_aes_block_array);
    cudaFree(cuda_r);
    cudaFree(cuda_output);
    cudaFree(cuda_scw);
    cudaFree(cuda_tcw_0);
    cudaFree(cuda_tcw_1);
}

void fss_dpf_evaluate(InputByteRelatedValuesEval cpu_eval_values, InputByteRelatedValuesGen cpu_values, aes_eval_block * cpu_aes_block_array, bool party, int input_length, int parallel){
    int lambda = 127;
    int input_byte = ceil(input_length/8);  
    
    BYTE key[240];
    int keyLen = 16;
    int blockLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
    prepare_key(cuda_key_block, key, keyLen); 

    int thrdperblock, num_sm;
    init_sm_thrd(num_sm, thrdperblock, parallel);
    std::cout << num_sm << " " << thrdperblock <<  std::endl;

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
    
    cudaMalloc(&cuda_reveal, parallel * input_byte * sizeof(uint8_t));
    cudaMemcpy(cuda_reveal, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // correction words, scw.shape = [parallel, input_length, LAMBDA_BYTE]
    cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t));
    cudaMemcpy(cuda_scw, cpu_values.scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // tcw.shape = [parallel, input_length]
    cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool));
    cudaMemcpy(cuda_tcw_0, cpu_values.tcw[0], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool));
    cudaMemcpy(cuda_tcw_1, cpu_values.tcw[1], parallel * input_length * sizeof(bool), cudaMemcpyHostToDevice);
    // output.shape = [parallel, input_byte]
    cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t));
    cudaMemcpy(cuda_output, cpu_values.output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_result, parallel * input_byte * sizeof(uint8_t));
    cudaMalloc(&cuda_dpf_eval, parallel*sizeof(class FssDpfEval));
    cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_eval_block));
    cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_eval_block), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    eval_init<<<BlockperGrid, ThreadperBlock>>>(cuda_dpf_eval, party, parallel);
    for(int i = 0; i < input_length; i++){
        AES_Encrypt_Eval<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, 176, parallel);
        st_init_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_scw, cuda_dpf_eval, i, input_byte, input_length, parallel);
        st_update_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_reveal, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_eval, i, input_byte, input_length, parallel);  
    }
    result_update_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_result, cuda_aes_block_array, cuda_output, cuda_dpf_eval, input_byte, parallel);
    cudaMemcpy(cpu_eval_values.result, cuda_result, parallel*input_byte*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_eval);
    cudaFree(cuda_key_block);
    cudaFree(cuda_reveal);
    cudaFree(cuda_result);
    cudaFree(cuda_scw);
    cudaFree(cuda_tcw_0);
    cudaFree(cuda_tcw_1);
    cudaFree(cuda_output);
    cudaFree(cuda_aes_block_array);
}

void fss_dpf_generate(InputByteRelatedValuesGen cpu_values, aes_gen_block * cpu_aes_block_array, int input_length, int parallel){
    int lambda = 127;  
    int input_byte = ceil(input_length/8);  
    BYTE key[240];
    int keyLen = 16;
    int blockLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, 2 * sizeof(class KeyBlock));
    prepare_key(cuda_key_block, key, keyLen);

    // printGpuBytes<<<1,1>>>(cuda_key_block->cuda_key[0], 0, 240);
    int thrdperblock, num_sm;
    init_sm_thrd(num_sm, thrdperblock, parallel);
    std::cout << num_sm << " " << thrdperblock <<  std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);


    FssDpfGen * cuda_dpf_gen;   
    aes_gen_block * cuda_aes_block_array;
    
    // input length related values
    // random values, shape = [parallel, input_byte]
    uint8_t * cuda_r;
    cudaMalloc(&cuda_r, parallel * input_byte * sizeof(uint8_t));
    cudaMemcpy(cuda_r, cpu_values.r, parallel * input_byte * sizeof(uint8_t), cudaMemcpyHostToDevice);
    // correction words, scw.shape = [parallel, input_length, input_byte]
    uint8_t * cuda_scw;
    cudaMalloc(&cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t));
    // tcw.shape = [parallel, input_length]
    bool * cuda_tcw_0;
    bool * cuda_tcw_1;
    cudaMalloc(&cuda_tcw_0, parallel * input_length * sizeof(bool));
    cudaMalloc(&cuda_tcw_1, parallel * input_length * sizeof(bool));
    // output.shape = [parallel, input_byte]
    uint8_t * cuda_output;
    cudaMalloc(&cuda_output, parallel * input_byte * sizeof(uint8_t));
    cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen));
    cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_gen_block));
    cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_gen_block), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    gen_init<<<BlockperGrid, ThreadperBlock>>>(cuda_dpf_gen, parallel);
    for(int i = 0; i < input_length; i++){        
        for(int j = 0; j < 2; j++){
            AES_Encrypt_Gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, 176, j, parallel);
            st_copy_gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_dpf_gen, j, parallel); 
        }       
        cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, input_byte, input_length, parallel);        
        for(int b = 0; b < 2; b++){
            st_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, b, input_byte, input_length, parallel);
        }
    }
    final_cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_output, cuda_dpf_gen, input_byte, parallel);
    cudaMemcpy(cpu_values.scw, cuda_scw, parallel * input_length * LAMBDA_BYTE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.tcw[0], cuda_tcw_0, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.tcw[1], cuda_tcw_1, parallel * input_length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_values.output, cuda_output, parallel * input_byte * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_gen);
    cudaFree(cuda_aes_block_array);
    cudaFree(cuda_r);
    cudaFree(cuda_output);
    cudaFree(cuda_scw);
    cudaFree(cuda_tcw_0);
    cudaFree(cuda_tcw_1);
}

void fss_dpf_evaluate(InputByteRelatedValuesEval cpu_eval_values, InputByteRelatedValuesGen cpu_values, aes_eval_block * cpu_aes_block_array, bool party, int input_length, int parallel){