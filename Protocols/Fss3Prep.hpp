/*
 * Fss3Prep.hpp
 *
 */

#include "Fss3Prep.h"
#include "GPU/gpu.h"

#ifndef PROTOCOLS_FSS3PREP_HPP_
#define PROTOCOLS_FSS3PREP_HPP_


template<class T>
void Fss3Prep<T>::gen_fake_dcf(int beta, int lambda)
{
   // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = int(lambda/8);
    std::cout << "lambda_bytes is " << lambda_bytes <<  std::endl;
    PRNG prng;
    prng.InitSeed();
    fstream k0, k1, r0, r1, r2;
    k0.open("Player-Data/2-fss/k0", ios::out);
    k1.open("Player-Data/2-fss/k1", ios::out);
    r0.open("Player-Data/2-fss/r0", ios::out);
    r1.open("Player-Data/2-fss/r1", ios::out);
    r2.open("Player-Data/2-fss/r2", ios::out);
    octet seed[2][2*(lambda_bytes*2+1)], convert_seed[2][lambda_bytes];    
    bigint s[2][2], v[2][2],  t[2][2], tmp_t[2], convert[2], tcw[2], a, scw, vcw, va, tmp[2], tmp_out, random_val;
    prng.InitSeed();
    prng.get(tmp[0], lambda);
    k0 << tmp[0] << " ";
    prng.get(tmp[1], lambda);
    k1 << tmp[1] << " ";
    prng.get(a, lambda);
    prng.get(random_val, lambda);
    r1 << a - random_val << " ";
    r0 << random_val << " ";
    r2 << random_val << " ";
    r0.close();
    r1.close();
    r2.close();
    tmp_t[0] = 0;
    tmp_t[1] = 1;
    int keep, lose;
    va = 0;


    //We can optimize keep into one bit here
    // generate the correlated word!
    for(int i = 0; i < lambda - 1; i++){
        keep = bigint(a >> lambda - i - 1).get_ui() & 1;
        lose = 1^keep;
        int n = 0;
        for(int j = 0; j < 2; j++){     
            // k is used for left and right
            bytesFromBigint(&seed[j][0], tmp[j], 2*(2*lambda_bytes+1));

            encryptwrapper(&seed[j][0], 2*(2*lambda_bytes+1), j);
            bigintFromBytes(t[0][j], &seed[j][0],1);
            t[0][j].get_mpz_t()->_mp_d[0] = t[0][j].get_mpz_t()->_mp_d[0]%2;
            bigintFromBytes(v[0][j], &seed[j][1],lambda_bytes);
            bigintFromBytes(s[0][j], &seed[j][lambda_bytes+1],lambda_bytes);

            bigintFromBytes(t[1][j], &seed[j][2*lambda_bytes+1],1);
            t[1][j].get_mpz_t()->_mp_d[0] = t[1][j].get_mpz_t()->_mp_d[0]%2;
            bigintFromBytes(v[1][j], &seed[j][2*lambda_bytes+2],lambda_bytes);
            bigintFromBytes(s[1][j], &seed[j][3*lambda_bytes+2],lambda_bytes);
            
        }
        scw = s[lose][0] ^ s[lose][1]; 
        std::cout << "scw is " << scw << std::endl;

        bytesFromBigint(&convert_seed[0][0], v[lose][0], lambda_bytes);
        encryptwrapper(&convert_seed[0][0], lambda_bytes, 1);
        bigintFromBytes(convert[0], &convert_seed[0][0], lambda_bytes);

        bytesFromBigint(&convert_seed[1][0], v[lose][1], lambda_bytes);
        encryptwrapper(&convert_seed[1][0], lambda_bytes, 1);
        bigintFromBytes(convert[1], &convert_seed[1][0], lambda_bytes);

        if(tmp_t[1])
            vcw = convert[0] + va - convert[1];
        else
            vcw = convert[1] - convert[0] - va;
        if(keep)
            vcw = vcw + tmp_t[1] * (-beta) + (1-tmp_t[1]) * beta;
        
        bytesFromBigint(&convert_seed[0][0], v[keep][0], lambda_bytes);
        encryptwrapper(&convert_seed[0][0], lambda_bytes, 1);
        bigintFromBytes(convert[0], &convert_seed[0][0], lambda_bytes);

        bytesFromBigint(&convert_seed[1][0], v[keep][1], lambda_bytes);
        encryptwrapper(&convert_seed[1][0], lambda_bytes, 1);
        bigintFromBytes(convert[1], &convert_seed[1][0], lambda_bytes);

        va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1-tmp_t[1]) * vcw;

        tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
        tcw[1] = t[1][0] ^ t[1][1] ^ keep;
        k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        
        tmp[0] = s[keep][0] ^ (tmp_t[0] * scw);
        tmp[1] =  s[keep][1] ^ (tmp_t[1] * scw);

        tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
        tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
    }
    
    encryptwrapper(&convert_seed[0][0], lambda_bytes, 1);
    bigintFromBytes(convert[0], &convert_seed[0][0], lambda_bytes);
    // prng.get(convert[0], lambda);

    encryptwrapper(&convert_seed[1][0], lambda_bytes, 1);
    bigintFromBytes(convert[1], &convert_seed[1][0], lambda_bytes);


    bytesFromBigint(&seed[0][0],  s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
    bytesFromBigint(&seed[1][0],  s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
    // prng.get(convert[1], lambda);
    k0 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k1 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k0.close();
    k1.close();

    return;
}

template<class T>
void Fss3Prep<T>::gen_fake_multi_spline_dcf(SubProcessor<T> &processor, int beta, int lambda, int base, int length){
   // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = max(16, (lambda + 1)/8);
    PRNG prng;
    prng.InitSeed();
    fstream k0, k1, r0, r1, r2;
    k0.open("Player-Data/2-fss/multi_k0", ios::out);
    k1.open("Player-Data/2-fss/multi_k1", ios::out);
    r0.open("Player-Data/2-fss/multi_r0", ios::out);
    r1.open("Player-Data/2-fss/multi_r1", ios::out);
    r2.open("Player-Data/2-fss/multi_r2", ios::out);
    octet seed[2][16];    
    bigint s[2][2], v[2][2],  t[2][2], tmp_t[2], convert[2], tcw[2], a, scw, vcw, va, tmp, tmp1, tmp_out;
    prng.InitSeed();
    prng.get(tmp, lambda);
    bytesFromBigint(&seed[0][0], tmp, lambda_bytes);
    k0 << tmp << " ";
    prng.get(tmp1, lambda);
    bytesFromBigint(&seed[1][0], tmp1, lambda_bytes);
    k1 << tmp1 << " ";
    
    prng.get(a, lambda);
    prng.get(tmp, lambda);
    r1 << a - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    r2.close();
    tmp_t[0] = 0;
    tmp_t[1] = 1;
    int keep, lose;
    va = 0;
    //We can optimize keep into one bit here
    // generate the correlated word!
    for(int i = 0; i < lambda - 1; i++){
        keep = bigint(a >> lambda - i - 1).get_ui() & 1;
        lose = 1^keep;
        for(int j = 0; j < 2; j++){     
            prng.SetSeed(seed[j]);
            // k is used for left and right
            for(int k = 0; k < 2; k++){
                prng.get(t[k][j], 1);
                prng.get(v[k][j], lambda);
                prng.get(s[k][j] ,lambda);
            }
        }
        scw = s[lose][0] ^ s[lose][1]; 
        // save convert(v0_lose) into convert[0]
        bytesFromBigint(&seed[0][0], v[lose][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], lambda);     
        // save convert(v1_lose) into convert[1]
        bytesFromBigint(&seed[0][0], v[lose][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], lambda);
        if(tmp_t[1])
            vcw = convert[0] + va - convert[1];
        else
            vcw = convert[1] - convert[0] - va;
        //keep == 1, lose = 0，so lose = LEFT
        if(keep)
            vcw = vcw + tmp_t[1]*(-beta) + (1-tmp_t[1]) * beta;
        // save convert(v0_keep) into convert[0]
        bytesFromBigint(&seed[0][0], v[keep][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], lambda);
        // save convert(v1_keep) into convert[1]
        bytesFromBigint(&seed[0][0], v[keep][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], lambda);
        
        va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1-tmp_t[1]) * vcw;
        tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
        tcw[1] = t[1][0] ^ t[1][1] ^ keep;
        k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        bytesFromBigint(&seed[0][0],  s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
        bytesFromBigint(&seed[1][0],  s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
        bigintFromBytes(tmp_out, &seed[0][0], 16);
        bigintFromBytes(tmp_out, &seed[1][0], 16);
        tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
        tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
    }
    
    prng.SetSeed(seed[0]);
    prng.get(convert[0], lambda);
    prng.SetSeed(seed[1]);
    prng.get(convert[1], lambda);
    k0 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k1 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    

    std::cout << "a is " << a << std::endl;
    typename T::clear rin, p_tmp, p_prev, z_0, z = 0;
    auto size = a.get_mpz_t()->_mp_size;
    mpn_copyi((mp_limb_t*)rin.get_ptr(), a.get_mpz_t()->_mp_d, abs(size));
    if(size < 0)
        rin = -rin;
    std::cout << "rin in " << rin << std::endl;
    
    
    std::cout << "reading splines: " << std::endl;
    for(int i = 1; i < length; i++){
        z = 0;
        p_prev = processor.C[base+i-1] + rin;
        p_tmp = processor.C[base+i] + rin;
        // std::cout << p_prev << " " << p_tmp << " " << processor.C[base+i-1] << " " << p_prev << " " << processor.C[base+i] << " " << p_tmp << std::endl;
        if(lambda == 128){
            // std::cout << " p > q " << (p_prev - p_tmp).get_bit(lambda) << " ap > p " << (processor.C[base+i-1] - p_prev).get_bit(lambda)  << " aq > q " << (processor.C[base+i] - p_tmp).get_bit(lambda) << std::endl; 
            z = (p_prev - p_tmp).get_bit(lambda) + (processor.C[base+i-1] - p_prev).get_bit(lambda) + (processor.C[base+i] - p_tmp).get_bit(lambda);
        }
        else{
            // std::cout << " p > q " << (p_prev - p_tmp).get_bit(lambda-1) << " ap > p " << (processor.C[base+i-1] - p_prev).get_bit(lambda-1)  << " aq > q " << (processor.C[base+i] - p_tmp).get_bit(lambda-1) << std::endl; 
            z = (p_prev - p_tmp).get_bit(lambda - 1) + (processor.C[base+i-1] - p_prev).get_bit(lambda - 1) + (processor.C[base+i] - p_tmp).get_bit(lambda - 1);
        }

        prng.get(tmp, lambda);
        auto size = tmp.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)z_0.get_ptr(), tmp.get_mpz_t()->_mp_d, abs(size));
        if(size < 0)
            z_0 = -z_0;
        // std::cout << "rin in " << rin << std::endl;
        r0 << z - z_0 << " ";
        r1 << z_0 << " ";
        // std::cout << " z is " << z << "z_0 is " << z_0 << std::endl;
    }
    k0.close();
    k1.close();
    r0.close();
    r1.close();
    
    
    
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

#endif /* PROTOCOLS_FSS3PREP_HPP_ */