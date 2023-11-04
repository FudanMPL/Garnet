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


__global__ void cw_update_gen(RandomValueBlock * cuda_r_block, CorrectionWord * cuda_cw, FssDpfGen * cuda_dpf_gen, int i, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_gen[global_thread_index].keep = ((cuda_r_block[global_thread_index].r[idx]) >> (7 - (i - (idx) * 8)))%2;
        cuda_dpf_gen[global_thread_index].lose = cuda_dpf_gen[global_thread_index].keep ^ 1;
        _xor(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][1], cuda_cw[global_thread_index].scw[i], LAMBDA_BYTE); 
        cuda_cw[global_thread_index].tcw[i][0] = cuda_dpf_gen[global_thread_index].t[0][0] ^ cuda_dpf_gen[global_thread_index].t[0][1] ^ cuda_dpf_gen[global_thread_index].keep ^ 1;
        cuda_cw[global_thread_index].tcw[i][1] = cuda_dpf_gen[global_thread_index].t[1][0] ^ cuda_dpf_gen[global_thread_index].t[1][1] ^ cuda_dpf_gen[global_thread_index].keep;
    }
    __syncthreads();
    return;
}


__global__ void st_update_gen(aes_gen_block * cuda_aes_block_array, CorrectionWord * cuda_cw, FssDpfGen * cuda_dpf_gen, int i, int b, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        //這裏有可能有錯
        _restricted_multiply(cuda_dpf_gen[global_thread_index].pre_t[b], cuda_cw[global_thread_index].scw[i] ,cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        _xor(cuda_aes_block_array[global_thread_index].block[b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], LAMBDA_BYTE);
        _double_copy(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ (cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_cw[global_thread_index].tcw[i][cuda_dpf_gen[global_thread_index].keep]);
    }
    __syncthreads();
    return;
}

__global__ void final_cw_update_gen(aes_gen_block * cuda_aes_block_array, CorrectionWord * cuda_cw, FssDpfGen * cuda_dpf_gen, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], LAMBDA_BYTE, INPUT_BYTE);
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], LAMBDA_BYTE, INPUT_BYTE);
        //假定beta是1
        if(cuda_dpf_gen[global_thread_index].pre_t[1]){
            _add(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], 1, INPUT_BYTE);  
            _sub(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_cw[global_thread_index].output, INPUT_BYTE, INPUT_BYTE);
        }
        else{
            _sub(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_cw[global_thread_index].output, INPUT_BYTE, INPUT_BYTE);
            _add(cuda_cw[global_thread_index].output, 1, INPUT_BYTE);                                                                                                                                                                                                                                                                                                                    
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

__global__ void st_init_eval(aes_eval_block * cuda_aes_block_array, CorrectionWord * cuda_cw, FssDpfEval * cuda_dpf_eval, int i, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, cuda_cw[global_thread_index].scw[i], cuda_cw[global_thread_index].scw[i], LAMBDA_BYTE);    
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = (cuda_dpf_eval[global_thread_index].s[0][LAMBDA_BYTE-1] % 2);
        cuda_dpf_eval[global_thread_index].t[1] = (cuda_dpf_eval[global_thread_index].s[1][LAMBDA_BYTE-1] % 2);
        _xor(cuda_dpf_eval[global_thread_index].s[0], cuda_cw[global_thread_index].scw[i], cuda_dpf_eval[global_thread_index].s[0], LAMBDA_BYTE);
        _xor(cuda_dpf_eval[global_thread_index].s[1], cuda_cw[global_thread_index].scw[i], cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE);
    }
    __syncthreads();
    return;
}

__global__ void st_update_eval(aes_eval_block * cuda_aes_block_array, RevealValueBlock * cuda_reveal_block, CorrectionWord * cuda_cw, FssDpfEval * cuda_dpf_eval, int i, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_eval[global_thread_index].xi = ((cuda_reveal_block[global_thread_index].reveal_val[idx]) >> (7 - (i - (idx) * 8)))%2;
        _double_copy(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], cuda_aes_block_array[global_thread_index].block, LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = cuda_dpf_eval[global_thread_index].t[0] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_cw[global_thread_index].tcw[i][0]);
        cuda_dpf_eval[global_thread_index].t[1] = cuda_dpf_eval[global_thread_index].t[1] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_cw[global_thread_index].tcw[i][1]);
        cuda_dpf_eval[global_thread_index].pre_t = cuda_dpf_eval[global_thread_index].t[cuda_dpf_eval[global_thread_index].xi];
    }
    __syncthreads();   
    return;
}
__global__ void result_update_eval(ResultBlock * cuda_res, aes_eval_block * cuda_aes_block_array, CorrectionWord * cuda_cw, FssDpfEval * cuda_dpf_eval, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, cuda_cw[global_thread_index].output, cuda_cw[global_thread_index].output, INPUT_BYTE);
        _convert(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], cuda_res[global_thread_index].result , LAMBDA_BYTE, INPUT_BYTE);
        _add(cuda_res[global_thread_index].result, cuda_cw[global_thread_index].output, cuda_res[global_thread_index].result, INPUT_BYTE, INPUT_BYTE);
    }
    __syncthreads();   
    return;  
}



