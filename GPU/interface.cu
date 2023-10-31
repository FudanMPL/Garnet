#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"

void fss_dpf_generate(RandomValueBlock * cpu_r_block, aes_block * cpu_aes_block_array[2], CorrectionWord * cpu_cw, int parallel){
    int lambda = 127;
    int bit_length = INPUT_BYTE * 8;

    
    BYTE key[240];
    int keyLen = 16;
    int blockLen = 16;
    KeyBlock * cuda_key_block;
    cudaMalloc(&cuda_key_block, sizeof(class KeyBlock));
    int expandKeyLen = AES_ExpandKey(key, keyLen);
    for(int i = 0; i < keyLen; i++){
        key[i] = i;
    }
    AES_ExpandKey(key, keyLen);
    cudaMemcpy(cuda_key_block->cuda_key[0], key, 240, cudaMemcpyHostToDevice);    
    for (int i = 0; i < keyLen; i++){
        key[i] = key[i] * 2;
    }
    AES_ExpandKey(key, keyLen);
    cudaMemcpy(cuda_key_block->cuda_key[1], key, 240, cudaMemcpyHostToDevice); 

    printGpuBytes<<<1,1>>>(cuda_key_block->cuda_key[0], 0, 240);
    int thrdperblock, num_sm, thrdperblock_l, num_sm_l;
    init_sm_thrd(num_sm, thrdperblock, num_sm_l, thrdperblock_l, parallel);
    std::cout << num_sm << " " << thrdperblock << " " << num_sm_l << " " << thrdperblock_l << std::endl;

    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);
    dim3 ThreadperBlockL(thrdperblock_l);
    dim3 BlockperGridL(num_sm_l);

    CorrectionWord * cuda_cw;   
    FssDpfGen * cuda_dpf_gen;   
    RandomValueBlock * cuda_r_block;   
    aes_block * cuda_aes_block_array[2];
    cudaMalloc(&cuda_cw, parallel*sizeof(class CorrectionWord));
    cudaMalloc(&cuda_dpf_gen, parallel*sizeof(class FssDpfGen));
    cudaMalloc(&cuda_r_block, parallel*sizeof(class RandomValueBlock));
    cudaMemcpy(cuda_r_block, cpu_r_block, parallel*sizeof(class RandomValueBlock), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_aes_block_array[0], parallel*sizeof(class aes_block));
    cudaMalloc(&cuda_aes_block_array[1], parallel*sizeof(class aes_block));
    cudaMemcpy(cuda_aes_block_array[0], cpu_aes_block_array[0], parallel*sizeof(class aes_block), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_aes_block_array[1], cpu_aes_block_array[1], parallel*sizeof(class aes_block), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
   
    // std::cout << "copy finished!" << std::endl;
    //记录加密算法开始时间
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    // int count;
 
    // cudaGetDeviceCount(&count);
    // printf("gpu num %d\n", count);
    // cudaGetDeviceProperties(&prop, 0);
    // printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    // printf("max grid dimensions: %d, %d, %d)\n",
    // prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    


    for(int i = 0; i < bit_length; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                s_copy_gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array[j], cuda_dpf_gen, k, j, parallel);
                AES_Encrypt<<<BlockperGrid,ThreadperBlock>>>(cuda_dpf_gen, cuda_key_block->cuda_key[k], k, j, 176, parallel);
                t_copy_gen<<<BlockperGrid,ThreadperBlock>>>(cuda_aes_block_array[j], cuda_dpf_gen, k, j, parallel);
                // printGpuBytes<<<1,1>>>(cuda_dpf_gen[0].s[k][j], 0, LAMBDA_BYTE);
            }
        }
        cw_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_r_block, cuda_cw, cuda_dpf_gen, i, parallel);
        printGpuBytes<<<1,1>>>(cuda_cw[0].scw[i], 0, LAMBDA_BYTE);
        for(int b = 0; b < 2; b++){
            st_update_gen<<<BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array[b], cuda_cw, cuda_dpf_gen, i, b, parallel);
        }
    }

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    // cudaFree(cuda_dpf_gen);
    // cudaFree(cuda_key_block);
    // cudaFree(cuda_cw);
    // cudaFree(cuda_gen_block);
}