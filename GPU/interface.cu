#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"
#include "test.cu"

void fss_dpf_generate(RandomValueBlock * cpu_r_block, aes_gen_block * cpu_aes_block_array, CorrectionWord * cpu_cw, int parallel){
    int lambda = 127;
    int bit_length = INPUT_BYTE * 8;

    
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


    CorrectionWord * cuda_cw;   
    FssDpfGen * cuda_dpf_gen;   
    RandomValueBlock * cuda_r_block;   
    aes_gen_block * cuda_aes_block_array;
    cudaMalloc(&cuda_cw, parallel*sizeof(class CorrectionWord));
    cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen));
    cudaMalloc(&cuda_r_block, parallel*sizeof(class RandomValueBlock));
    cudaMemcpy(cuda_r_block, cpu_r_block, parallel*sizeof(class RandomValueBlock), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_gen_block));
    cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_gen_block), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    gen_init<<<BlockperGrid, ThreadperBlock>>>(cuda_dpf_gen, parallel);
    for(int i = 0; i < bit_length; i++){        
        for(int j = 0; j < 2; j++){
            AES_Encrypt_Gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, 176, j, parallel);
            st_copy_gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_dpf_gen, j, parallel); 
        }
        cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_r_block, cuda_cw, cuda_dpf_gen, i, parallel);        
        
        for(int b = 0; b < 2; b++){
            st_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_cw, cuda_dpf_gen, i, b, parallel);
        }
    }
    final_cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_cw, cuda_dpf_gen, parallel);

    cudaMemcpy(cpu_cw, cuda_cw, parallel*sizeof(class CorrectionWord), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_gen);
    cudaFree(cuda_key_block);
    cudaFree(cuda_cw);
    cudaFree(cuda_r_block);
    cudaFree(cuda_aes_block_array);
}


void fss_dpf_evaluate(RevealValueBlock * cpu_reveal, aes_eval_block * cpu_aes_block_array, CorrectionWord * cpu_cw, ResultBlock * cpu_res, bool party, int parallel){
    int lambda = 127;
    int bit_length = INPUT_BYTE * 8;

    
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

    CorrectionWord * cuda_cw;   
    FssDpfEval * cuda_dpf_eval;   
    RevealValueBlock * cuda_reveal_block;   
    aes_eval_block * cuda_aes_block_array;
    ResultBlock * cuda_res;
    cudaMalloc(&cuda_cw, parallel*sizeof(class CorrectionWord));
    cudaMemcpy(cuda_cw, cpu_cw, parallel*sizeof(class CorrectionWord), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_dpf_eval, parallel*sizeof(class FssDpfEval));
    cudaMalloc(&cuda_reveal_block, parallel*sizeof(class RevealValueBlock));
    cudaMemcpy(cuda_reveal_block, cpu_reveal, parallel*sizeof(class RevealValueBlock), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_aes_block_array, parallel*sizeof(class aes_eval_block));
    cudaMemcpy(cuda_aes_block_array, cpu_aes_block_array, parallel*sizeof(class aes_eval_block), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_res, parallel*sizeof(class ResultBlock));

    cudaDeviceSynchronize();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    eval_init<<<BlockperGrid, ThreadperBlock>>>(cuda_dpf_eval, party, parallel);
    for(int i = 0; i < bit_length; i++){
        AES_Encrypt_Eval<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array, cuda_key_block, 176, parallel);
        test<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_dpf_eval, parallel);
        st_init_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_cw, cuda_dpf_eval, i, parallel);
        st_update_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_reveal_block, cuda_cw, cuda_dpf_eval, i, parallel);  
    }
    result_update_eval<<<BlockperGrid, ThreadperBlock>>>(cuda_res, cuda_aes_block_array, cuda_cw, cuda_dpf_eval, parallel);
    cudaMemcpy(cpu_res, cuda_res, parallel*sizeof(class ResultBlock), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_eval);
    cudaFree(cuda_key_block);
    cudaFree(cuda_cw);
    cudaFree(cuda_reveal_block);
    cudaFree(cuda_aes_block_array);
}