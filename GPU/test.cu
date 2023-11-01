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
    printf("t[0] is %d, t[1] is %d\n", cuda_dpf_eval[0].t[0], cuda_dpf_eval[0].t[1]);
    printf("tcw[0] is %d, tcw[1] is %d\n", cuda_cw[0].tcw[i][0], cuda_cw[0].tcw[i][1]);
    printf("pre_t is %d\n",cuda_dpf_eval[0].pre_t);
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

