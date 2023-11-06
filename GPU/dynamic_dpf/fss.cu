#include "aes.cu"
#include "utils.cu"

__host__ void prepare_key(KeyBlock * cuda_key_block, BYTE * key, int keyLen){
    int expandKeyLen = AES_ExpandKey(key, keyLen);
    for(int i = 0; i < keyLen; i++){
        key[i] = i;
    }
    AES_ExpandKey(key, keyLen);
    cudaMemcpy(cuda_key_block[0].cuda_key, key, 240, cudaMemcpyHostToDevice);    
    for (int i = 0; i < keyLen; i++){
        key[i] = key[i] * 2;
    }
    AES_ExpandKey(key, keyLen);
    cudaMemcpy(cuda_key_block[1].cuda_key, key, 240, cudaMemcpyHostToDevice); 
}

__host__ void init_sm_thrd(int &num_sm, int &thrdperblock, int parallel){
    //3080最多的线程支持数是1024
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    num_sm = prop.multiProcessorCount; 
    thrdperblock = int(parallel/num_sm);
    if(parallel%num_sm>0)
        thrdperblock++;
    if(thrdperblock>1024){
        thrdperblock = 1024;
        num_sm = parallel/1024;
        if(parallel%1024>0){
            num_sm++;
        }
    }
    return;
}



__global__ void gen_init(FssDpfGen * cuda_dpf_gen, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        for(int i = 0; i < 2; i++)
            cuda_dpf_gen[global_thread_index].pre_t[i] = i;
    }
    __syncthreads();
    return;
}

__global__ void st_copy_gen(aes_gen_block * cuda_aes_block_array, FssDpfGen * cuda_dpf_gen, int j, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        for(int k = 0; k < 2; k++){
            _copy(cuda_aes_block_array[global_thread_index].block[j], cuda_dpf_gen[global_thread_index].s[k][j], k*LAMBDA_BYTE, 0, LAMBDA_BYTE);
            cuda_dpf_gen[global_thread_index].t[k][j] = cuda_dpf_gen[global_thread_index].s[k][j][LAMBDA_BYTE - 1] % 2;
        }
    }
    __syncthreads();
    return;
}


__global__ void cw_update_gen(uint8_t * cuda_r, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int input_byte, int input_length, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_gen[global_thread_index].keep = ((cuda_r[global_thread_index * input_byte + idx]) >> (input_byte - 1 - (i - (idx) * 8)))%2;
        cuda_dpf_gen[global_thread_index].lose = cuda_dpf_gen[global_thread_index].keep ^ 1;
        _xor(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][1], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], LAMBDA_BYTE); 
        cuda_tcw_0[global_thread_index * input_length + i] = cuda_dpf_gen[global_thread_index].t[0][0] ^ cuda_dpf_gen[global_thread_index].t[0][1] ^ cuda_dpf_gen[global_thread_index].keep ^ 1;
        cuda_tcw_1[global_thread_index * input_length + i] = cuda_dpf_gen[global_thread_index].t[1][0] ^ cuda_dpf_gen[global_thread_index].t[1][1] ^ cuda_dpf_gen[global_thread_index].keep;
    }
    __syncthreads();
    return;
}

__global__ void st_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int b, int input_byte, int input_length, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        //這裏有可能有錯
        _restricted_multiply(cuda_dpf_gen[global_thread_index].pre_t[b], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE] ,cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        _xor(cuda_aes_block_array[global_thread_index].block[b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], LAMBDA_BYTE);
        _double_copy(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        if(cuda_dpf_gen[global_thread_index].keep)
            cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ (cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_tcw_1[global_thread_index * input_length * 1 + i * 1]);
        else
            cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ (cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_tcw_0[global_thread_index * input_length * 1 + i * 1]);
    }
    __syncthreads();
    return;
}

__global__ void final_cw_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfGen * cuda_dpf_gen, int input_byte, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], LAMBDA_BYTE, input_byte);
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], LAMBDA_BYTE, input_byte);
        //假定beta是1
        if(cuda_dpf_gen[global_thread_index].pre_t[1]){
            _add(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], 1, input_byte);  
            _sub(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], &cuda_output[global_thread_index * input_byte], input_byte, input_byte);
        }
        else{
            _sub(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], &cuda_output[global_thread_index * input_byte], input_byte, input_byte);
            _add(&cuda_output[global_thread_index * input_byte], 1, input_byte);                                                                                                                                                                                                                                                                                                                    
        }
    }
    __syncthreads();   
    return;
}

__global__ void eval_init(FssDpfEval * cuda_dpf_eval, bool party,int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        cuda_dpf_eval[global_thread_index].pre_t = party;
    }
    __syncthreads();
    return;
}

__global__ void test(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_scw, FssDpfEval * cuda_dpf_eval, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
    }
    __syncthreads();
    return;
}

__global__ void st_init_eval(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_scw, FssDpfEval * cuda_dpf_eval, int i, int input_byte, int input_length, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], LAMBDA_BYTE);    
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = (cuda_dpf_eval[global_thread_index].s[0][LAMBDA_BYTE-1] % 2);
        cuda_dpf_eval[global_thread_index].t[1] = (cuda_dpf_eval[global_thread_index].s[1][LAMBDA_BYTE-1] % 2);
        _xor(cuda_dpf_eval[global_thread_index].s[0], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE] , cuda_dpf_eval[global_thread_index].s[0], LAMBDA_BYTE);
        _xor(cuda_dpf_eval[global_thread_index].s[1], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE] , cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE);
    }
    __syncthreads();
    return;
}

__global__ void st_update_eval(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_reveal, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfEval * cuda_dpf_eval, int i, int input_byte, int input_length, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_eval[global_thread_index].xi = ((cuda_reveal[global_thread_index * input_byte + idx]) >> (7 - (i - (idx) * 8)))%2;
        _double_copy(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], cuda_aes_block_array[global_thread_index].block, LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = cuda_dpf_eval[global_thread_index].t[0] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_tcw_0[global_thread_index * input_length + i]);
        cuda_dpf_eval[global_thread_index].t[1] = cuda_dpf_eval[global_thread_index].t[1] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_tcw_1[global_thread_index * input_length + i]);
        cuda_dpf_eval[global_thread_index].pre_t = cuda_dpf_eval[global_thread_index].t[cuda_dpf_eval[global_thread_index].xi];
    }
    __syncthreads();   
    return;
}
__global__ void result_update_eval(uint8_t * cuda_result, aes_eval_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfEval * cuda_dpf_eval, int input_byte, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, &cuda_output[global_thread_index * input_byte], &cuda_output[global_thread_index * input_byte], input_byte);
        _convert(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], &cuda_result[global_thread_index * input_byte] , LAMBDA_BYTE, input_byte);
        _add(&cuda_result[global_thread_index * input_byte], &cuda_output[global_thread_index * input_byte], &cuda_result[global_thread_index * input_byte], input_byte, input_byte);
    }
    __syncthreads();   
    return;  
}



