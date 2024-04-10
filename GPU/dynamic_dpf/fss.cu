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

__host__ void init_sm_thrd(int &num_sm, int &thrdperblock, int parallel, int &rounds){
    //3080最多的线程支持数是1024
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    num_sm = prop.multiProcessorCount; 
    thrdperblock = int(parallel/num_sm);
    if(parallel%num_sm>0)
        thrdperblock++;
    if(thrdperblock>256){
        thrdperblock = 256;
        num_sm = parallel/256;
        if(parallel%256>0){
            num_sm++;
        }
    }
    rounds = ceil(num_sm/prop.multiProcessorCount);
    num_sm = prop.multiProcessorCount;
    return;
}



__device__ void gen_init(FssDpfGen * cuda_dpf_gen, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        for(int i = 0; i < 2; i++)
            cuda_dpf_gen[global_thread_index].pre_t[i] = i;
    }
    return;
}

__device__ void st_copy_gen(aes_gen_block * cuda_aes_block_array, FssDpfGen * cuda_dpf_gen, int j, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        for(int k = 0; k < 2; k++){
            _copy(cuda_aes_block_array[global_thread_index].block[j], cuda_dpf_gen[global_thread_index].s[k][j], k*LAMBDA_BYTE, 0, LAMBDA_BYTE);
            cuda_dpf_gen[global_thread_index].t[k][j] = cuda_dpf_gen[global_thread_index].s[k][j][LAMBDA_BYTE - 1] % 2;
        }
    }
    return;
}


__device__ void cw_update_gen(uint8_t * cuda_r, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int input_byte, int input_length, int global_thread_index, int parallel){
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_gen[global_thread_index].keep = ((cuda_r[global_thread_index * input_byte + idx]) >> (input_byte - 1 - (i - (idx) * 8)))%2;
        cuda_dpf_gen[global_thread_index].lose = cuda_dpf_gen[global_thread_index].keep ^ 1;
        _xor(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].lose][1], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], LAMBDA_BYTE); 
        cuda_tcw_0[global_thread_index * input_length + i] = cuda_dpf_gen[global_thread_index].t[0][0] ^ cuda_dpf_gen[global_thread_index].t[0][1] ^ cuda_dpf_gen[global_thread_index].keep ^ 1;
        cuda_tcw_1[global_thread_index * input_length + i] = cuda_dpf_gen[global_thread_index].t[1][0] ^ cuda_dpf_gen[global_thread_index].t[1][1] ^ cuda_dpf_gen[global_thread_index].keep;
    }
    return;
}

__device__ void st_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfGen * cuda_dpf_gen, int i, int b, int input_byte, int input_length, int global_thread_index, int parallel){
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
    return;
}

__device__ void final_cw_update_gen(aes_gen_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfGen * cuda_dpf_gen, int input_byte, int global_thread_index, int parallel){
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
    return;
}

__device__ void final_cw_update_gen_compress(uint8_t * cuda_r, aes_gen_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfGen * cuda_dpf_gen, int input_byte, int compress, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][0], LAMBDA_BYTE, input_byte);
        _convert(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][1], LAMBDA_BYTE, input_byte);
        //计算select_vector哪一个对应下标处为1 
        //最多压缩64位，8个比特构成的数
        int select_idx = ((cuda_r[global_thread_index * input_byte + input_byte - 1] << (8 - compress)) >> (8 - compress));
        cuda_output[global_thread_index * input_byte + input_byte - 1 - int(select_idx / 8)] = (uint8_t(1) << (select_idx % 8 - 1));
        //注释掉这两行就能测试beta的正确性
        _add(cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][cuda_dpf_gen[global_thread_index].pre_t[1] ^ 1], &cuda_output[global_thread_index * input_byte], &cuda_output[global_thread_index * input_byte], input_byte ,input_byte);  
        _sub(&cuda_output[global_thread_index * input_byte], cuda_dpf_gen[global_thread_index].s[cuda_dpf_gen[global_thread_index].keep][cuda_dpf_gen[global_thread_index].pre_t[1]], &cuda_output[global_thread_index * input_byte], input_byte, input_byte);

    }
    return;
}


__device__ void eval_init(FssDpfEval * cuda_dpf_eval, bool party, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        cuda_dpf_eval[global_thread_index].pre_t = party;
    }
    return;
}

__device__ void test(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_scw, FssDpfEval * cuda_dpf_eval, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
    }
    return;
}

__device__ void st_init_eval(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_scw, FssDpfEval * cuda_dpf_eval, int i, int input_byte, int input_length, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE], LAMBDA_BYTE);    
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[0], 0, 0 ,LAMBDA_BYTE);
        _copy(cuda_aes_block_array[global_thread_index].block, cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE, 0 ,LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = (cuda_dpf_eval[global_thread_index].s[0][LAMBDA_BYTE-1] % 2);
        cuda_dpf_eval[global_thread_index].t[1] = (cuda_dpf_eval[global_thread_index].s[1][LAMBDA_BYTE-1] % 2);
        _xor(cuda_dpf_eval[global_thread_index].s[0], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE] , cuda_dpf_eval[global_thread_index].s[0], LAMBDA_BYTE);
        _xor(cuda_dpf_eval[global_thread_index].s[1], &cuda_scw[global_thread_index * input_length * LAMBDA_BYTE + i * LAMBDA_BYTE] , cuda_dpf_eval[global_thread_index].s[1], LAMBDA_BYTE);
    }
    return;
}

__device__ void st_update_eval(aes_eval_block * cuda_aes_block_array, uint8_t * cuda_reveal, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, FssDpfEval * cuda_dpf_eval, int i, int input_byte, int input_length, int global_thread_index, int parallel){
    int idx = int(i/8);
    if(global_thread_index < parallel){
        cuda_dpf_eval[global_thread_index].xi = ((cuda_reveal[global_thread_index * input_byte + idx]) >> (input_byte - 1 - (i - (idx) * 8)))%2;
        _double_copy(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], cuda_aes_block_array[global_thread_index].block, LAMBDA_BYTE);
        cuda_dpf_eval[global_thread_index].t[0] = cuda_dpf_eval[global_thread_index].t[0] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_tcw_0[global_thread_index * input_length + i]);
        cuda_dpf_eval[global_thread_index].t[1] = cuda_dpf_eval[global_thread_index].t[1] ^ (cuda_dpf_eval[global_thread_index].pre_t * cuda_tcw_1[global_thread_index * input_length + i]);
        cuda_dpf_eval[global_thread_index].pre_t = cuda_dpf_eval[global_thread_index].t[cuda_dpf_eval[global_thread_index].xi];
    }
    return;
}

__device__ void result_update_eval(uint8_t * cuda_result, aes_eval_block * cuda_aes_block_array, uint8_t * cuda_output, FssDpfEval * cuda_dpf_eval, int input_byte, int global_thread_index, int parallel){
    if(global_thread_index < parallel){
        _restricted_multiply(cuda_dpf_eval[global_thread_index].pre_t, &cuda_output[global_thread_index * input_byte], &cuda_output[global_thread_index * input_byte], input_byte);
        _convert(cuda_dpf_eval[global_thread_index].s[cuda_dpf_eval[global_thread_index].xi], &cuda_result[global_thread_index * input_byte] , LAMBDA_BYTE, input_byte);
        _add(&cuda_result[global_thread_index * input_byte], &cuda_output[global_thread_index * input_byte], &cuda_result[global_thread_index * input_byte], input_byte, input_byte);
    }
    return;  
}


__global__ void dpf_gen(aes_gen_block * cuda_aes_block_array, KeyBlock * cuda_key_block, FssDpfGen * cuda_dpf_gen, uint8_t * cuda_r, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, uint8_t * cuda_output, int input_length, int input_byte, int parallel, int num_sm, int thrdperblock, int round){
    int global_thread_index = round * num_sm * thrdperblock + blockDim.x * blockIdx.x + threadIdx.x;
    gen_init(cuda_dpf_gen, global_thread_index, parallel);
    for(int i = 0; i < input_length; i++){   
        for(int j = 0; j < 2; j++){
            __shared__ BYTE AES_ShiftRowTab[16];
            __shared__ BYTE AES_Sbox[256];
            __shared__ BYTE AES_ShiftRowTab_Inv[16];
            __shared__ BYTE AES_Sbox_Inv[256];
            __shared__ BYTE AES_xtime[256];
            if(global_thread_index < parallel){
                if(threadIdx.x == 0 ){
                    AES_Init(AES_Sbox, AES_ShiftRowTab, AES_Sbox_Inv, AES_xtime, AES_ShiftRowTab_Inv);
                }
                __syncthreads();
                BYTE block[16]; 
                

                for(int k = 0; k < 2; k++){
                    for(int i=0; i<16; i++){
                        block[i] = cuda_aes_block_array[global_thread_index].block[j][k*16+i];
                    }
                    int l = 176, i;
                    //printBytes(block, 16);
                    AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[0]);
                    for(i = 16; i < l - 16; i += 16) {
                        AES_SubBytes(block, AES_Sbox);
                        AES_ShiftRows(block, AES_ShiftRowTab);
                        AES_MixColumns(block, AES_xtime);
                        AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                    }
                    AES_SubBytes(block, AES_Sbox);
                    AES_ShiftRows(block, AES_ShiftRowTab);
                    AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                    for(int i=0; i<16; i++){
                        cuda_aes_block_array[global_thread_index].block[j][k*16+i] = block[i];
                    }
                }
            }
            st_copy_gen(cuda_aes_block_array, cuda_dpf_gen, j, global_thread_index, parallel); 
        }       
        cw_update_gen(cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, input_byte, input_length, global_thread_index, parallel);        
        for(int b = 0; b < 2; b++){
            st_update_gen(cuda_aes_block_array, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, b, input_byte, input_length, global_thread_index, parallel);
        }
    }
    final_cw_update_gen(cuda_aes_block_array, cuda_output, cuda_dpf_gen, input_byte, global_thread_index, parallel);
    return;
}

__global__ void dpf_eval(aes_eval_block * cuda_aes_block_array, KeyBlock * cuda_key_block, FssDpfEval * cuda_dpf_eval, uint8_t * cuda_reveal, uint8_t * cuda_scw, uint8_t * cuda_result, bool * cuda_tcw_0, bool * cuda_tcw_1, uint8_t * cuda_output, int input_length, int input_byte, int party, int parallel, int round, int num_sm, int thrdperblock){
    int global_thread_index = round * num_sm * thrdperblock + blockDim.x * blockIdx.x + threadIdx.x;

    eval_init(cuda_dpf_eval, party, global_thread_index, parallel);
    for(int i = 0; i < input_length; i++){
        // AES_Encrypt_Eval(cuda_aes_block_array, cuda_key_block, 176, threadIdx.x, global_thread_index, parallel);
        __shared__ BYTE AES_ShiftRowTab[16];
        __shared__ BYTE AES_Sbox[256];
        __shared__ BYTE AES_ShiftRowTab_Inv[16];
        __shared__ BYTE AES_Sbox_Inv[256];
        __shared__ BYTE AES_xtime[256];
        if(global_thread_index < parallel){
            if(threadIdx.x == 0){
                AES_Init(AES_Sbox, AES_ShiftRowTab, AES_Sbox_Inv, AES_xtime, AES_ShiftRowTab_Inv);
            }
            __syncthreads();
            BYTE block[16];
            for(int k = 0; k < 2; k++){
                for(int i=0; i<16; i++){
                    block[i] = cuda_aes_block_array[global_thread_index].block[k*16+i];
                }
                int l = 176, i;
                AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[0]);
                for(i = 16; i < l - 16; i += 16) {
                    AES_SubBytes(block, AES_Sbox);
                    AES_ShiftRows(block, AES_ShiftRowTab);
                    AES_MixColumns(block, AES_xtime);
                    AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                }
                AES_SubBytes(block, AES_Sbox);
                AES_ShiftRows(block, AES_ShiftRowTab);
                AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                for(int i=0; i<16; i++){
                    cuda_aes_block_array[global_thread_index].block[k*16+i] = block[i];
                }
            }
        }
        st_init_eval(cuda_aes_block_array, cuda_scw, cuda_dpf_eval, i, input_byte, input_length, global_thread_index, parallel);
        st_update_eval(cuda_aes_block_array, cuda_reveal, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_eval, i, input_byte, input_length, global_thread_index, parallel);     
    }
    result_update_eval(cuda_result, cuda_aes_block_array, cuda_output, cuda_dpf_eval, input_byte, global_thread_index, parallel);
    return;
}


__global__ void dpf_compress_gen(aes_gen_block * cuda_aes_block_array, KeyBlock * cuda_key_block, FssDpfGen * cuda_dpf_gen, uint8_t * cuda_r, uint8_t * cuda_scw, bool * cuda_tcw_0, bool * cuda_tcw_1, uint8_t * cuda_output, int input_length, int input_byte, int compress, int parallel, int num_sm, int thrdperblock, int round){
    int global_thread_index = round * num_sm * thrdperblock + blockDim.x * blockIdx.x + threadIdx.x;
    gen_init(cuda_dpf_gen, global_thread_index, parallel);
    for(int i = 0; i < input_length; i++){        
        for(int j = 0; j < 2; j++){
            __shared__ BYTE AES_ShiftRowTab[16];
            __shared__ BYTE AES_Sbox[256];
            __shared__ BYTE AES_ShiftRowTab_Inv[16];
            __shared__ BYTE AES_Sbox_Inv[256];
            __shared__ BYTE AES_xtime[256];
            if(global_thread_index < parallel){
                if(threadIdx.x == 0 ){
                    AES_Init(AES_Sbox, AES_ShiftRowTab, AES_Sbox_Inv, AES_xtime, AES_ShiftRowTab_Inv);
                }
                __syncthreads();
                BYTE block[16]; 
                

                for(int k = 0; k < 2; k++){
                    for(int i=0; i<16; i++){
                        block[i] = cuda_aes_block_array[global_thread_index].block[j][k*16+i];
                    }
                    int l = 176, i;
                    //printBytes(block, 16);
                    AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[0]);
                    for(i = 16; i < l - 16; i += 16) {
                        AES_SubBytes(block, AES_Sbox);
                        AES_ShiftRows(block, AES_ShiftRowTab);
                        AES_MixColumns(block, AES_xtime);
                        AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                    }
                    AES_SubBytes(block, AES_Sbox);
                    AES_ShiftRows(block, AES_ShiftRowTab);
                    AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
                    for(int i=0; i<16; i++){
                        cuda_aes_block_array[global_thread_index].block[j][k*16+i] = block[i];
                    }
                }
            }
            st_copy_gen(cuda_aes_block_array, cuda_dpf_gen, j, global_thread_index, parallel); 
        }       
        cw_update_gen(cuda_r, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, input_byte, input_length, global_thread_index, parallel);        
        for(int b = 0; b < 2; b++){
            st_update_gen(cuda_aes_block_array, cuda_scw, cuda_tcw_0, cuda_tcw_1, cuda_dpf_gen, i, b, input_byte, input_length, global_thread_index, parallel);
        }
    }
    final_cw_update_gen_compress(cuda_r, cuda_aes_block_array, cuda_output, cuda_dpf_gen, input_byte, compress, global_thread_index, parallel);
    
}
