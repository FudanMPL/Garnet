#include "aes.cu"
#include "utils.cu"

__global__ void test(FssDpfGen * cuda_dpf_gen, CorrectionWord * cuda_cw, int b, int i){
    // printf("ri is %d\n", cuda_dpf_gen[0].keep);
    // printf("t[0][0] is %d, t[0][1] is %d, t[1][0] is %d, t[1][1] is %d\n", cuda_dpf_gen[0].t[0][0], cuda_dpf_gen[0].t[0][1], cuda_dpf_gen[0].t[1][0], cuda_dpf_gen[0].t[1][1]);
    // printf("tcw[0] is %d, tcw[1] is %d\n", cuda_cw[0].tcw[i][0], cuda_cw[0].tcw[i][1]);
    printf("pre_t[0] is %d, pre_t[1] is %d\n",cuda_dpf_gen[0].pre_t[0], cuda_dpf_gen[0].pre_t[1]);
}

__global__ void test(FssDpfEval * cuda_dpf_eval, CorrectionWord * cuda_cw, int i){
    printf("xi is %d\n", cuda_dpf_eval[0].xi);
    // printf("t[0] is %d, t[1] is %d\n", cuda_dpf_eval[0].t[0], cuda_dpf_eval[0].t[1]);
    // printf("tcw[0] is %d, tcw[1] is %d\n", cuda_cw[0].tcw[i][0], cuda_cw[0].tcw[i][1]);
    printf("pre_t is %d\n",cuda_dpf_eval[0].pre_t);
    printGpuBytes<<<1,1>>>(cuda_dpf_eval[0].s[cuda_dpf_eval[0].xi], 0, LAMBDA_BYTE);
}

__global__ void test(aes_eval_block * cuda_aes_block_array, FssDpfEval * cuda_dpf_eval, int parallel){
     int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = cuda_dpf_eval[global_thread_index].s[0][LAMBDA_BYTE-1] % 2;
        cuda_dpf_eval[global_thread_index].t[1] = cuda_dpf_eval[global_thread_index].s[1][LAMBDA_BYTE-1] % 2;
    }
    __syncthreads();
}

__global__ void test_cw_update_gen(uint8_t * cuda_r, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int input_byte, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_gen[global_thread_index].keep = ((cuda_r[global_thread_index * (input_byte) + idx]) >> (input_byte - 1 - (i - (idx) * 8)))%2;
        cuda_dpf_gen[global_thread_index].lose = cuda_dpf_gen[global_thread_index].keep ^ 1;
        _xor(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][1], &cuda_scw[global_thread_index * input_byte * LAMBDA_BYTE + i * LAMBDA_BYTE], LAMBDA_BYTE); 
        cuda_tcw_0[global_thread_index * (input_byte * 8) + i] = cuda_dpf_gen[global_thread_index].t[0][0] ^ cuda_dpf_gen[global_thread_index].t[0][1] ^ cuda_dpf_gen[global_thread_index].keep ^ 1;
        cuda_tcw_1[global_thread_index * (input_byte * 8) + i] = cuda_dpf_gen[global_thread_index].t[1][0] ^ cuda_dpf_gen[global_thread_index].t[1][1] ^ cuda_dpf_gen[global_thread_index].keep;
    }
    __syncthreads();
    return;
}

__global__ void test_st_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int b, int input_byte, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        //這裏有可能有錯
        _restricted_multiply(cuda_dpf_gen[global_thread_index].pre_t[b], &cuda_scw[global_thread_index * input_byte * LAMBDA_BYTE + i * LAMBDA_BYTE] ,cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        _xor(cuda_aes_block_array[global_thread_index].block[b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], LAMBDA_BYTE);
        _double_copy(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_aes_block_array[global_thread_index].block[b], LAMBDA_BYTE);
        if(cuda_dpf_gen[global_thread_index].keep)
            cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ (cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_tcw_1[global_thread_index * (input_byte * 8) * 1 + i * 1]);
        else
            cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ (cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_tcw_0[global_thread_index * (input_byte * 8) * 1 + i * 1]);
    }
    __syncthreads();
    return;
}

__global__ void test_final_cw_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfGen * cuda_dpf_gen, int input_byte, int parallel){
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