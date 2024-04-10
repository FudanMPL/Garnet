/*
 * Fss3Prep.hpp
 *
 */

#include "Fss3Prep.h"
#include "GPU/dynamic_dpf/gpu.h"
#include <cstdlib>
#include <string>
#include <ctime>
#ifndef PROTOCOLS_FSS3PREP_HPP_
#define PROTOCOLS_FSS3PREP_HPP_

template<class T>
void Fss3Prep<T>::gen_dpf_correction_word(Player& P, int gen_num){
    octetStream cs;
    int eval_num_0 = (gen_num + 1) % 3;
    int eval_num_1 = (gen_num + 2) % 3;
    std::cout << "Entered Fss3Prep gen_dpf_correction_word" << std::endl;
    if(P.my_num()== gen_num){
        vector<bigint> test;
        test.push_back(bigint(0));
        this->get_correction_word_no_count(DATA_DPF);
        test[0].pack(cs);
        P.send_to(eval_num_0, cs);
        test[0].pack(cs);
        P.send_to(eval_num_1, cs);
    }
    else{
        P.receive_player(gen_num, cs);
        vector <bigint> test(100,1);
        test[0].unpack(cs);
        // this->fss_dpf_eval_values.r_share.unpack(cs)
        std::cout << "Fss3Prep.hpp recieved!" << std::endl;
        std::cout << test.size() << std::endl;
        
    }
    return;
}

template<class T>
void Fss3Prep<T>::gen_fake_dcf(int beta, int n)
{
   // lambda is the security parameter, which is a preset value.
    int lambda = 127, lambda_bytes = 16, ring_size = int(n/8);
    // std::cout << "ring size is " << ring_size << std::endl;
    PRNG prng;
    prng.InitSeed();
    fstream k0, k1, r0, r1, r2;
    k0.open("Player-Data/2-fss/k0", ios::out);
    k1.open("Player-Data/2-fss/k1", ios::out);
    r0.open("Player-Data/2-fss/r0", ios::out);
    r1.open("Player-Data/2-fss/r1", ios::out);
    r2.open("Player-Data/2-fss/r2", ios::out);
    octet seed[2][lambda_bytes];    
    bigint s[2][2], v[2][2], tmp_t[2], convert[2], scw, vcw, va, tmp, tmp1, tmp_out, x1, y1, r_out, msbx1, r_in, r_in_2, u, w, z;
    bool t[2][2], tcw[2];
    
    // initialize r_in and r_out
    prng.get(r_in, n);
    prng.get(r_out, n);

    // set max_val = 1<<n
    bigint max_val = 1;
    max_val = max_val << n;

    // x1 = 2^n - r_in
    x1 = max_val - r_in;
    std::cout << "x1 is " << x1 << std::endl;

    // get msb of x1
    if(r_in == 0){
        msbx1 = 0;
    }
    else{
        msbx1 = (x1 >> (n-1));
    }
    std::cout << "msbx1 is " << msbx1 << std::endl;
    // y1 = x1 mod 1 << (n-1)
    if(msbx1)
        y1 = x1 - (max_val >> 1);
    else
        y1 = x1;
    if(r_in == 0){
        msbx1 = 0;
        y1 = 0;
    }
    std::cout << "y1 is " << y1 << std::endl;
    std::cout << "r_in is " << r_in << std::endl;
    // intialize seed
    prng.InitSeed();
    prng.get(tmp, lambda);
    prng.get(tmp1, lambda);
    bytesFromBigint(&seed[0][0], tmp, lambda_bytes);
    bytesFromBigint(&seed[1][0], tmp1, lambda_bytes);
    k0 << tmp << " ";
    k1 << tmp1 << " ";
    // initialize alpha
    
    prng.get(tmp, n);
    prng.get(tmp1, n);

    r1 << r_in - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    
    r1 << (msbx1 ^ 1) - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";

    // generate random values for select protocol
    prng.get(r_in, 1);
    r_in_2 = 0;
    u = r_in;
    w = u * r_in_2;
    z = 2 * u * r_in_2;

    prng.get(tmp, 1);
    r1 << (r_in ^ tmp) << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    prng.get(tmp, n);
    r1 << u - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    prng.get(tmp, n);
    r1 << r_in_2 - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    prng.get(tmp, n);
    r1 << w - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    prng.get(tmp, n);
    r1 << z - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    

    r0.close();
    r1.close();
    r2.close();
    tmp_t[0] = 0;
    tmp_t[1] = 1;
    int keep, lose;
    va = 0;
    
    //We can optimize keep into one bit here
    // generate the correlated word!
    
    for(int i = 1; i < n-16; i++){
        keep = bigint(y1 >> n - i - 1).get_ui() & 1;
        lose = 1^keep;
        for(int j = 0; j < 2; j++){     
            prng.SetSeed(seed[j]);
            // k is used for left and right
            for(int k = 0; k < 2; k++){
                prng.get(v[k][j], lambda);
                prng.get(s[k][j] ,lambda);
                t[k][j] = s[k][j].get_ui() & 1;
            }
        }
        scw = s[lose][0] ^ s[lose][1]; 
        // save bigint(Convert v0_lose) into convert[0]
        if(n <= 128){
            convert[0] = v[lose][0] >> (lambda-n);
            convert[1] = v[lose][1] >> (lambda-n);
        }
        if(tmp_t[1])
            vcw = convert[0] + va - convert[1];
        else
            vcw = convert[1] - convert[0] - va;
        //keep == 1, lose = 0，so lose = LEFT
        if(keep)
            vcw = vcw + tmp_t[1]*(-beta) + (1-tmp_t[1]) * beta;
        // save bigintConvert(v0_keep) into convert[0]
        if(n <= 128){
            convert[0] = v[keep][0] >> (lambda-n);
            convert[1] = v[keep][1] >> (lambda-n);
        }
        va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1-tmp_t[1]) * vcw;
        tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
        tcw[1] = t[1][0] ^ t[1][1] ^ keep;
        // std::cout << "scw, vcw, tcw0, tcw1 are " << scw << " " << vcw << " "<< tcw[0] << " " << tcw[1] << std::endl;
        k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        bytesFromBigint(&seed[0][0],  s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
        bytesFromBigint(&seed[1][0],  s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
        tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
        tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
    }
    if(n <= 128){
        convert[0] = (s[keep][0] ^ (tmp_t[0] * scw)) >> (lambda-n);
        convert[1] = (s[keep][1] ^ (tmp_t[1] * scw)) >> (lambda-n);
    }
    k0 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k1 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k0.close();
    k1.close();

    return;
}


template<class T>
void Fss3Prep<T>::buffer_dpf_with_gpu(int lambda){
    int bit_length = T::clear::MAX_N_BITS;
    int input_byte = ceil(bit_length/8.0);
    int batch_size = OnlineOptions::singleton.batch_size;
    PRNG prng;
    bigint seed[2], r, r0, r1, res0, res1;
    prng.InitSeed();
    for(int i = 0; i < batch_size; i++){
        prng.get(r, bit_length);
        prng.get(r0, bit_length);
        r1 = r - r0;
        // r = 0;
        prng.get(seed[0], lambda);
        prng.get(seed[1], lambda);

        bytesFromBigint(&this->fss_dpf_gen_values.r[i * input_byte], r, input_byte);
        bytesFromBigint(&this->fss_dpf_gen_values.r_share_0[i * input_byte], r0, input_byte);
        bytesFromBigint(&this->fss_dpf_gen_values.r_share_1[i * input_byte], r1, input_byte);
        bytesFromBigint(&this->fss_dpf_gen_seeds[i].block[0][0], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&this->fss_dpf_gen_seeds[i].block[0][LAMBDA_BYTE], seed[0], LAMBDA_BYTE);
        bytesFromBigint(&this->fss_dpf_gen_seeds[i].block[1][0], seed[1], LAMBDA_BYTE);
        bytesFromBigint(&this->fss_dpf_gen_seeds[i].block[1][LAMBDA_BYTE], seed[1], LAMBDA_BYTE);
    }
    // std::cout << "fss_dpf_generate calling" << std::endl;
    fss_dpf_generate(this->fss_dpf_gen_values, this->fss_dpf_gen_seeds, bit_length, batch_size);
    return;
}

template<class T>
void Fss3Prep<T>::buffer_dabits(ThreadQueues*)
{
    assert(this->protocol);
    assert(this->proc);

    typedef typename T::bit_type BT;
    int n_blocks = DIV_CEIL(this->buffer_size, BT::default_length);
    int n_bits = n_blocks * BT::default_length;

    vector<BT> b(n_blocks);

    vector<array<T, 3>> a(n_bits);
    Player& P = this->proc->P;

    for (int i = 0; i < 2; i++)
    {
        for (auto& x : b)
            x[i].randomize(this->protocol->shared_prngs[i]);

        int j = P.get_offset(i);

        for (int k = 0; k < n_bits; k++)
            a[k][j][i] = b[k / BT::default_length][i].get_bit(
                    k % BT::default_length);
    }

    // the first multiplication
    vector<T> first(n_bits), second(n_bits);
    typename T::Input input(P);

    if (P.my_num() == 0)
    {
        for (auto& x : a)
            input.add_mine(x[0][0] * x[1][1]);
    }
    else
        input.add_other(0);

    input.exchange();

    for (int k = 0; k < n_bits; k++)
        first[k] = a[k][0] + a[k][1] - 2 * input.finalize(0);

    input.reset_all(P);

    if (P.my_num() != 0)
    {
        for (int k = 0; k < n_bits; k++)
            input.add_mine(first[k].local_mul(a[k][2]));
    }

    input.add_other(1);
    input.add_other(2);
    input.exchange();

    for (int k = 0; k < n_bits; k++)
    {
        second[k] = first[k] + a[k][2]
                - 2 * (input.finalize(1) + input.finalize(2));
        this->dabits.push_back({second[k],
            b[k / BT::default_length].get_bit(k % BT::default_length)});
    }
}

template<class T>
void Fss3Prep<T>::get_correction_word_no_count(Dtype dtype){
    if(dtype == DATA_DPF){   
        std::cout << "dpf cnt is " << dpf_cnt << std::endl;
        if(this->dpf_cnt == 0){
            buffer_dpf_with_gpu(127);
        }
        int input_byte = ceil(T::clear::MAX_N_BITS/8.0);
        bigint r100;
        bigintFromBytes(r100, &this->fss_dpf_gen_values.r[100 * input_byte], input_byte);
        std::cout << "r 100 is " << r100 << std::endl;
    }
}



#endif /* PROTOCOLS_FSS3PREP_HPP_ */