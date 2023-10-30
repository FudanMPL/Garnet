#include "aes.cu"
#include "utils.cu"

__global__ void init_seed_gen(FssDpfGenerateBlock * cuda_gen_block, FssDpfGen * cuda_dpf_gen, aes_block * cuda_aes_block_array, int i, int cur_group, int parallel, int num_sm, int thrdperblock){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    cuda_dpf_gen[cur_group].pre_t[i][global_thread_index] = i;
    _copy<<<1, LAMBDA_BYTE>>>(cuda_gen_block[cur_group].seed[i], cuda_aes_block_array[cur_group * MAX_PARALLEL + global_thread_index].block, global_thread_index * LAMBDA_BYTE, 0, LAMBDA_BYTE);
    __syncthreads();
    return;
}

// __global__ void st_copy_gen(aes_block * cuda_aes_block_array, FssDpfGen * cuda_dpf_gen, int k, int j, int parallel){
//     int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
//     __syncthreads();
//     if(global_thread_index < parallel){
//         cuda_dpf_gen.t[k][j][global_thread_index] = cuda_dpf_gen[global_thread_index].s[k][j][LAMBDA_BYTE - 1] % 2;
//     }
//     __syncthreads();
//     return;
// }

// __global__ void cw_update_gen(CorrectionWord * cuda_cw, FssDpfGen * cuda_dpf_gen, int lose ,int idx, int parallel){
//     int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
//     __syncthreads();
//     if(global_thread_index < parallel){
//         _xor<<<1,LAMBDA_BYTE>>>(cuda_dpf_gen[global_thread_index].s[lose][0], cuda_dpf_gen[global_thread_index].s[lose][1],cuda_cw[global_thread_index].scw[idx], LAMBDA_BYTE); 
//     }
//     __syncthreads();
//     return;
// }