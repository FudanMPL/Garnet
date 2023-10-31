#include "aes.cu"
#include "utils.cu"

__host__ void prepare_key(KeyBlock * cuda_key_block, BYTE * key, int keyLen){
    
}

__host__ void init_sm_thrd(int &num_sm, int &thrdperblock, int &num_sm_l, int &thrdperblock_l, int parallel){
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
    //此外，global函数里调用global函数（通常是用LAMBDA_BYTES个线程并行），因此还需要额外计算一个num_sm_l, thrdperblock_l
    num_sm_l = prop.multiProcessorCount; 
    thrdperblock_l = int(parallel/num_sm_l);
    if(parallel%num_sm_l>0)
        thrdperblock_l++;
    if(thrdperblock_l>128){
        thrdperblock_l = 128;
        num_sm_l = parallel/128;
        if(parallel%128>0){
            num_sm_l++;
        }
    }
    return;
}

__global__ void s_copy_gen(aes_block * cuda_aes_block_array, FssDpfGen * cuda_dpf_gen, int k, int j, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_gen[global_thread_index].s[k][j], 0, 0, LAMBDA_BYTE);
    }
    __syncthreads();
    return;
}

__global__ void t_copy_gen(aes_block * cuda_aes_block_array, FssDpfGen * cuda_dpf_gen, int k, int j, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    if(global_thread_index < parallel){
        cuda_dpf_gen[global_thread_index].t[k][j] = cuda_dpf_gen[global_thread_index].s[k][j][LAMBDA_BYTE - 1] % 2;
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

__global__ void st_update_gen(aes_block * cuda_aes_block_array, CorrectionWord * cuda_cw, FssDpfGen * cuda_dpf_gen, int i, int b, int parallel){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __syncthreads();
    _restricted_multiply(cuda_dpf_gen[global_thread_index].pre_t, cuda_cw[global_thread_index].scw[i] ,cuda_aes_block_array[global_thread_index].block, LAMBDA_BYTE);
    _xor(cuda_aes_block_array[global_thread_index].block, cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][b], cuda_aes_block_array[global_thread_index].block, LAMBDA_BYTE);
    cuda_dpf_gen[global_thread_index].pre_t[b] = cuda_dpf_gen[global_thread_index].t[cuda_dpf_gen[global_thread_index].keep][b] ^ cuda_dpf_gen[global_thread_index].pre_t[b] * cuda_cw[global_thread_index].tcw[i][cuda_dpf_gen[global_thread_index].keep];
    __syncthreads();
    return;
}