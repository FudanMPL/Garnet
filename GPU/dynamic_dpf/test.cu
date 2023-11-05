#include "aes.cu"
#include "utils.cu"

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
