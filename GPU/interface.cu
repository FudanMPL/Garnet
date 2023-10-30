#include "aes.cu"
#include "utils.cu"
#include "fss_struct.h"
#include "fss.cu"

void fss_dpf_generate(FssDpfGenerateBlock * gen_block_cpu, CorrectionWord * cw_cpu, int parallel, int create_size){
    int lambda = 127;
    int bit_length = INPUT_BYTE * 8;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    

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
    
    int thrdperblock;
    int num_sm = prop.multiProcessorCount; 
    if(parallel >= MAX_PARALLEL){
        thrdperblock = MAX_PARALLEL/num_sm;
        std::cout << MAX_PARALLEL << " " << num_sm << " " << MAX_PARALLEL % num_sm << std::endl;
        if(MAX_PARALLEL%num_sm>0)
            thrdperblock++;
        // if(thrdperblock>1024){
        //     thrdperblock = 1024;
        //     num_sm = MAX_PARALLEL/1024;
        //     if(MAX_PARALLEL%1024>0){
        //         num_sm++;
        //     }
        // }
    }
    else{
        int thrdperblock = parallel/num_sm;
        if(parallel%num_sm>0)
            thrdperblock++;
        if(thrdperblock>1024){
            thrdperblock = 1024;
            num_sm = parallel/1024;
            if(parallel%1024>0){
                num_sm++;
            }
        }
    }
    
    dim3 ThreadperBlock(thrdperblock);
    dim3 BlockperGrid(num_sm);

    CorrectionWord * cuda_cw;
    cudaMalloc((void**)&cuda_cw, create_size * sizeof(class CorrectionWord));
    
    FssDpfGen * cuda_dpf_gen;
    cudaMalloc(&cuda_dpf_gen, create_size * sizeof(class FssDpfGen));
    
    FssDpfGenerateBlock * cuda_gen_block;
    cudaMalloc(&cuda_gen_block, create_size * sizeof(class FssDpfGen));
    cudaMemcpy(cuda_gen_block, gen_block_cpu, create_size * sizeof(class FssDpfGenerateBlock), cudaMemcpyHostToDevice);

    aes_block * cuda_aes_block_array[2];
    cudaMalloc(&cuda_aes_block_array[0], parallel * sizeof(class aes_block));
    cudaMalloc(&cuda_aes_block_array[1], parallel * sizeof(class aes_block));

    // cudaDeviceSynchronize();
    std::cout << "copy finished!" << std::endl;
    //记录加密算法开始时间
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    std::cout << num_sm << " " << thrdperblock << std::endl; 

    int count;
 
    cudaGetDeviceCount(&count);
    printf("gpu num %d\n", count);
    cudaGetDeviceProperties(&prop, 0);
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    for(int cur_group = 0; cur_group < create_size; cur_group++){
        for(int i = 0; i < 2; i++){
            init_seed_gen<<<num_sm, thrdperblock>>>(cuda_gen_block, cuda_dpf_gen, cuda_aes_block_array[i], i, cur_group, parallel, num_sm, thrdperblock);
        }
    }
    for(int i = 0; i < bit_length; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                AES_Encrypt<<<num_sm, thrdperblock>>>(cuda_aes_block_array[j], cuda_key_block->cuda_key[k], 176, parallel);
    //             st_copy_gen<<<num_sm, thrdperblock>>>(cuda_aes_block_array, cuda_dpf_gen, k, j, parallel);
            }
        }
    }
    
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float msecTotal1,total;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    total=msecTotal1/1000;
    printf("eval time:%f\n",total);
    cudaFree(cuda_dpf_gen);
    cudaFree(cuda_key_block);
    cudaFree(cuda_cw);
    cudaFree(cuda_gen_block);
}