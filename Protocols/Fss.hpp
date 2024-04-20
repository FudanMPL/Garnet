/*
 * FSS.hpp
 *
 */

#ifndef PROTOCOLS_FSS_HPP_
#define PROTOCOLS_FSS_HPP_
#include <thread>
#include "Fss.h"
#include "Processor/Processor.h"
#include "Processor/TruncPrTuple.h"
#include "Tools/benchmarking.h"
#include "Tools/Bundle.h"

#include "ReplicatedInput.h"
#include "Fss3Share2k.h"

#include "ReplicatedPO.hpp"
#include "Math/Z2k.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

template<class T>
Fss<T>::Fss(Player& P) : ReplicatedBase(P)
{
    assert(T::vector_length == 2);
}

template<class T>
Fss<T>::Fss(const ReplicatedBase& other) :
        ReplicatedBase(other)
{
}

template<class T>
void Fss<T>::init_fss_cmp_prep(SubProcessor<T> &processor){
    bigint signal = 0;
    octetStream cs;
    if(this->fss3prep == nullptr){
        std::cout << "fss3prep is nullptr!" << std::endl;
        if(P.my_num() == GEN){  
            this->fss3prep->gen_fake_dcf(1, T::clear::MAX_N_BITS);
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        else{
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
    }
    this->fss3prep->init_offline_values(&processor, 0);
    return;
}

template<class T>
void Fss<T>::init_fss_conv_relu_prep(SubProcessor<T> &processor, int float_bits){
    bigint signal = 0;
    octetStream cs;
    if(this->fss3prep == nullptr){
        std::cout << "fss3prep is nullptr!" << std::endl;
        if(P.my_num() == GEN){  
            this->fss3prep->gen_fake_conv_relu(1, T::clear::MAX_N_BITS, float_bits);
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        else{
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
    }
    this->fss3prep->init_offline_values(&processor, 1);
    return;
}

template<class T>
void Fss<T>::init(Preprocessing<T>& prep, typename T::MAC_Check& MC)
{
    this->prep = &prep;
    this->MC = &MC;
}


template<class T>
void Fss<T>::distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    auto& args = instruction.get_start();
    int dcf_parallel = int(args.size()/args[0]);    
    typename T::open_type result[dcf_parallel];
    bigint res_bool[dcf_parallel], tmp_bool_res;
    fstream r;
    octetStream cs[2], reshare_cs, b_cs; //cs0, cs1; 
    MC->init_open(P, n);
    for(size_t i = 0; i < args.size(); i+= args[i]){ 
        // reveal x + r
        auto dest = &proc.S[args[i+3]][0];
        *dest = *dest + typename T::open_type(this->fss3prep->r_share);
        MC->prepare_open(proc.S[args[i+3]]);   
    }
    MC->exchange(P);

    if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
        int cnt = 0;
        bigint msb_xhat, tmp_cmp, tb[dcf_parallel], res;
        typename T::open_type tmp[dcf_parallel], tmp_sub = (1LL << (n-1));
        for(size_t t = 0; t < args.size(); t += args[t]){
            result[cnt] = MC->finalize_raw();
            // std::cout << "reveal result is " << result[cnt] << std::endl;            
            // get_bit begins from 0 to n-1
            msb_xhat = result[cnt].get_bit(n - 1);
            if(msb_xhat){
                result[cnt] = result[cnt] - (1LL << (n-1));
            }
            result[cnt] = tmp_sub - result[cnt] - 1;
            // std::cout << "evaluate is " << result[cnt] << std::endl;
            tb[cnt] = this->evaluate(result[cnt], n, 1, 0);
            tb[cnt] = bigint(typename T::open_type(tb[cnt])) ^ bigint(typename T::open_type(this->fss3prep->r_b)) ^ (P.my_num() * msb_xhat);
            std::cout << "res_bool[cnt] is " << tb[cnt] << std::endl;
            res_bool[cnt] = bigint(typename T::open_type(tb[cnt])).get_ui() % 2;
            std::cout << "res_bool[cnt] is " << res_bool[cnt] << std::endl;
            res_bool[cnt] = res_bool[cnt] ^ this->fss3prep->rs_b;
            res_bool[cnt].pack(b_cs);
        }
        P.send_to((P.my_num()^1), b_cs);
        P.receive_player((P.my_num()^1), b_cs);
        for(size_t t = 0; t < args.size(); t += args[t]){    
            tmp_bool_res.unpack(b_cs);
            res_bool[cnt] = res_bool[cnt] ^ tmp_bool_res;
            // std::cout << "res_bool[cnt] is " << res_bool[cnt] << std::endl;
            // std::cout << this->fss3prep->u << " " << this->fss3prep->r_in_2 << " " << this->fss3prep->w << " " << this->fss3prep->z << std::endl;
            if(res_bool[cnt])
                res = P.my_num()-this->fss3prep->u-this->fss3prep->r_in_2+this->fss3prep->w;
            else
                res = this->fss3prep->u +  this->fss3prep->w - this->fss3prep->z;
            proc.S[args[t+2]][0] =  bigint(typename T::open_type(P.my_num()-res));
        }
    }
    else{ 
        for(size_t i = 0; i < args.size(); i += args[i])
            proc.S[args[i+2]][0] = 0;
    }
    for(size_t i = 0; i < args.size(); i+= args[i]){
        proc.S[args[i+2]][0].pack(reshare_cs);
    }
    P.send_to((P.my_num()+1)%P.num_players(), reshare_cs);
    P.receive_player((P.my_num()+2)%P.num_players(), reshare_cs);
    for(size_t i = 0; i < args.size(); i+= args[i]){
        proc.S[args[i+2]][1].unpack(reshare_cs);   
    }
    return;
}


// T::clear 是错的
template<class T>
void Fss<T>::distributed_comparison_function_gpu(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    auto& args = instruction.get_start();
    int parallel = int(args.size()/args[0]);    
    int input_byte = int(T::clear::MAX_N_BITS/8);
    MC->init_open(P, n);
    bigint r_share_bigint;
    typename T::clear r_share;
    int cnt = 0, size = 0;
    for(size_t i = 0; i < args.size(); i+= args[i]){
        auto dest = &proc.S[args[i+3]][0];
        bigintFromBytes(r_share_bigint, &this->fss3prep->fss_dpf_eval_values.r_share[cnt * input_byte], input_byte);
        size = r_share_bigint.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)r_share.get_ptr(), r_share_bigint.get_mpz_t()->_mp_d, abs(size));
        *dest = *dest - r_share;
        MC->prepare_open(proc.S[args[i+3]]);  
        cnt += 1;
    }
    MC->exchange(P);
    bigint test;
    if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
        cnt = 0;
        typename T::clear shift[parallel];
        typename T::clear zero = 0;
        bigint x_add_r_bigint;
        // 存放x+r的数值
        uint8_t x_add_r[parallel];
        for(size_t t = 0; t < args.size(); t+= args[t]){
            shift[cnt] = MC->finalize_raw();
            size = int(input_byte / 8);
            mpn_copyi(x_add_r_bigint.get_mpz_t()->_mp_d, (mp_limb_t*)shift[cnt].get_ptr(), size);
            if(shift[cnt].get() >= zero.get())
                x_add_r_bigint.get_mpz_t()->_mp_size = -size;
            else
                x_add_r_bigint.get_mpz_t()->_mp_size = size;
            // std::cout << x_add_r_bigint << std::endl;
            // std::cout << x_add_r_bigint.get_mpz_t()->_mp_size << std::endl;
            // std::cout << "shift is " << shift[cnt] << std::endl;
            
            // fss_dpf_evaluate(this->fss3prep->fss_dpf_eval_values, this->fss3prep->fss_dpf_eval_seeds, P.my_num(), n, parallel, cnt, shift[cnt]);
            cnt += 1;
        }
    }
}

template <class T>
void Fss<T>::generate(){
    //判断buffer里面有没有数
    //有数值就读出即可
    //没有数就gen_dcp
}


template<class T>
bigint Fss<T>::evaluate(typename T::open_type x, int n, int result_length, int drop_least_bits){
    
    PRNG prng;
    prng.ReSeed();
    int b = P.my_num(), xi;
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda = 127, lambda_bytes = 16;
    
    octet seed[lambda_bytes];
    // r is the random value generate by GEN
    bigint s_hat[2], v_hat[2], s[2], v[2], convert[2], tmp_bigint, tmp_v, tmp_out;
    bool t_hat[2], t[2], tmp_t;
    
    bytesFromBigint(&seed[0], this->fss3prep->seed_bigint, lambda_bytes);
    tmp_t = b;
    tmp_v = 0;
    
    for(int i = 1 + drop_least_bits; i < n; i++){
        xi = x.get_bit(n - i - 1);
        
        prng.SetSeed(seed);
        for(int j = 0; j < 2; j++){
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j], lambda);
            t_hat[j] = s_hat[j].get_ui() & 1;
            s[j] = s_hat[j] ^ (tmp_t * this->fss3prep->scw[i-1]);
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->tcw[j][i-1]);
        }

        // std::cout << "v_hat[0], v_hat[1] are " << v_hat[0] << " " << v_hat[1] << std::endl;
        if(n <= 128){
            // std::cout << "length is " << lambda-n << std::endl;
            convert[0] = v_hat[0] >> (lambda-result_length);
            convert[1] = v_hat[1] >> (lambda-result_length);
        }

        // std::cout << "convert[0], convert[1] are " << convert[0] << " " << convert[1] << std::endl;
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]) + (1^b) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
        // std::cout << tmp_v << std::endl;
    }

    if(n <= 128)
        convert[0] = s[xi] >> (lambda-result_length);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * this->fss3prep->final_cw) + (1^b) * (convert[0] + tmp_t * this->fss3prep->final_cw);
    // std::cout << "final_tmp_v" << tmp_v << std::endl;
    return tmp_v;  
}

template<class T>
bigint Fss<T>::evaluate_conv_relu(typename T::open_type x, int n, int result_length){
    
    PRNG prng;
    prng.InitSeed();
    int b = P.my_num(), xi;
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda = 127, lambda_bytes = 16;
    
    octet seed[lambda_bytes];
    // r is the random value generate by GEN
    bigint s_hat[2], v_hat[2], s[2], v[2], convert[2], tmp_bigint, tmp_v, tmp_out;
    bool t_hat[2], t[2], tmp_t;
    bytesFromBigint(&seed[0], this->fss3prep->full_seed_bigint, lambda_bytes);
    std::cout << this->fss3prep->full_seed_bigint << std::endl;
    tmp_t = b;
    tmp_v = 0;
    for(int i = 1; i < n; i++){
        xi = x.get_bit(n - i - 1);
        
        prng.SetSeed(seed);
        for(int j = 0; j < 2; j++){
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j], lambda);
            t_hat[j] = s_hat[j].get_ui() & 1;
            s[j] = s_hat[j] ^ (tmp_t * this->fss3prep->full_scw[i-1]);
            std::cout << this->fss3prep->full_scw[i-1] << std::endl;
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->full_tcw[j][i-1]);
            std::cout << this->fss3prep->full_tcw[j][i-1] << std::endl;
        }

        // std::cout << "v_hat[0], v_hat[1] are " << v_hat[0] << " " << v_hat[1] << std::endl;
        if(n <= 128){
            // std::cout << "length is " << lambda-n << std::endl;
            convert[0] = v_hat[0] >> (lambda-result_length);
            convert[1] = v_hat[1] >> (lambda-result_length);
        }

        // std::cout << "convert[0], convert[1] are " << convert[0] << " " << convert[1] << std::endl;
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * this->fss3prep->full_vcw[i-1]) + (1^b) * (convert[xi] + tmp_t * this->fss3prep->full_vcw[i-1]);
        std::cout << this->fss3prep->full_vcw[i-1] << std::endl;
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
        // std::cout << tmp_v << std::endl;
    }

    if(n <= 128)
        convert[0] = s[xi] >> (lambda-result_length);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * this->fss3prep->full_final_cw) + (1^b) * (convert[0] + tmp_t * this->fss3prep->full_final_cw);
    std::cout << this->fss3prep->full_final_cw << std::endl;
    // std::cout << "final_tmp_v" << tmp_v << std::endl;
    return tmp_v;  
}

// codes for multiply without truncation and reshare
template <class T>
inline void Fss<T>::init_dotprod_without_trunc()
{
    add_shares.clear();
    dotprod_share.assign_zero();
}

template <class T>
inline void Fss<T>::prepare_dotprod_without_trunc(const T &x, const T &y)
{
    dotprod_share = dotprod_share.lazy_add(x.local_mul(y));
}

template <class T>
inline void Fss<T>::next_dotprod_without_trunc()
{   
    // dotprod_share.normalize();
    // prepare_reshare(dotprod_share);
    // dotprod_share.assign_zero();
    add_shares.push_back(dotprod_share);
    dotprod_share.assign_zero();
}

template <class T>
inline T Fss<T>::finalize_dotprod_without_trunc(int length)
{
    (void)length;
    this->dot_counter++;
    this->counter++;
    this->bit_counter+=length;
    T result;
    result[0] = add_shares.next();
    // std::cout<< "result[0] is " << result[0] << std::endl;
    result[1] = 0;
    // result[1].unpack(os[1], length);
    return result;
}


template <class T>
void Fss<T>::init_mul()
{
    for (auto &o : os)
        o.reset_write_head();
    add_shares.clear();
}

template <class T>
void Fss<T>::prepare_mul(const T &x,
                                const T &y, int n)
{
    typename T::value_type add_share = x.local_mul(y);
    prepare_reshare(add_share, n);
}

template <class T>
void Fss<T>::prepare_reshare(const typename T::clear &share,
                                    int n)
{
    typename T::value_type tmp[2];
    for (int i = 0; i < 2; i++)
        tmp[i].randomize(shared_prngs[i], n);
    auto add_share = share + tmp[0] - tmp[1];
    add_share.pack(os[0], n);
    add_shares.push_back(add_share);
}

template <class T>
void Fss<T>::exchange()
{
    if (os[0].get_length() > 0)
        P.pass_around(os[0], os[1], 1);
    this->rounds++;
}

template <class T>
void Fss<T>::start_exchange()
{
    P.send_relative(1, os[0]);
    this->rounds++;
}

template <class T>
void Fss<T>::stop_exchange()
{
    P.receive_relative(-1, os[1]);
}

template <class T>
inline T Fss<T>::finalize_mul(int n)
{
    this->counter++;
    this->bit_counter += n;
    T result;
    result[0] = add_shares.next();
    result[1].unpack(os[1], n);
    return result;
}

template <class T>
inline void Fss<T>::init_dotprod()
{
    init_mul();
    dotprod_share.assign_zero();
}

template <class T>
inline void Fss<T>::prepare_dotprod(const T &x, const T &y)
{
    dotprod_share = dotprod_share.lazy_add(x.local_mul(y));
}

template <class T>
inline void Fss<T>::next_dotprod()
{
    dotprod_share.normalize();
    prepare_reshare(dotprod_share);
    dotprod_share.assign_zero();
}

template <class T>
inline T Fss<T>::finalize_dotprod(int length)
{
    (void)length;
    this->dot_counter++;
    return finalize_mul();
}

template <class T>
T Fss<T>::get_random()
{
    T res;
    for (int i = 0; i < 2; i++)
        res[i].randomize(shared_prngs[i]);
    return res;
}


template <class T>
void Fss<T>::randoms(T &res, int n_bits)
{
    for (int i = 0; i < 2; i++)
        res[i].randomize_part(shared_prngs[i], n_bits);
}

template <class T>
template <class U>
void Fss<T>::trunc_pr(const vector<int> &regs, int size, U &proc,
                            false_type)
{
    assert(regs.size() % 4 == 0);
    assert(proc.P.num_players() == 3);
    assert(proc.Proc != 0);
    typedef typename T::clear value_type;
    int gen_player = 2;
    int comp_player = 1;
    bool generate = P.my_num() == gen_player;
    bool compute = P.my_num() == comp_player;
    ArgList<TruncPrTupleWithGap<value_type>> infos(regs);
    auto &S = proc.get_S();

    octetStream cs;
    ReplicatedInput<T> input(P);

    if (generate)
    {
        SeededPRNG G;
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto r = G.get<value_type>();
                input.add_mine(info.upper(r));
                if (info.small_gap())
                    input.add_mine(info.msb(r));
                (r + S[info.source_base + i][0]).pack(cs);
            }
        P.send_to(comp_player, cs);
    }
    else
        input.add_other(gen_player);

    if (compute)
    {
        P.receive_player(gen_player, cs);
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto c = cs.get<value_type>() + S[info.source_base + i].sum();
                input.add_mine(info.upper(c));
                if (info.small_gap())
                    input.add_mine(info.msb(c));
            }
    }

    input.add_other(comp_player);
    input.exchange();
    init_mul();

    for (auto info : infos)
        for (int i = 0; i < size; i++)
        {
            this->trunc_pr_counter++;
            auto c_prime = input.finalize(comp_player);
            auto r_prime = input.finalize(gen_player);
            S[info.dest_base + i] = c_prime - r_prime;

            if (info.small_gap())
            {
                auto c_dprime = input.finalize(comp_player);
                auto r_msb = input.finalize(gen_player);
                S[info.dest_base + i] += ((r_msb + c_dprime) << (info.k - info.m));
                prepare_mul(r_msb, c_dprime);
            }
        }
    exchange();

    for (auto info : infos)
        for (int i = 0; i < size; i++)
            if (info.small_gap())
                S[info.dest_base + i] -= finalize_mul()
                                        << (info.k - info.m + 1);
}

template <class T>
template <class U>
void Fss<T>::trunc_pr(const vector<int> &regs, int size, U &proc,
                            true_type)
{
    (void)regs, (void)size, (void)proc;
    throw runtime_error("trunc_pr not implemented");
}

template <class T>
template <class U>
void Fss<T>::trunc_pr(const vector<int> &regs, int size,
                            U &proc)
{
    this->trunc_rounds++;
    trunc_pr(regs, size, proc, T::clear::characteristic_two);
}

template <class T>
template <class U>
void Fss<T>::change_domain(const vector<int> &regs, U &proc)
{
    cout << regs.size() << " " << proc.S.size() << endl;
}

template <class T>
void Fss<T>::cisc(SubProcessor<T> &processor, const Instruction &instruction)
{   
    int r0 = instruction.get_r(0), n = T::clear::MAX_N_BITS;
    string tag((char *)&r0, 4);
    auto& args = instruction.get_start();
    int parallel = int(args.size()/args[0]);
    if (tag == string("LTZ\0", 4))
    {
        this->init_fss_cmp_prep(processor);
        if(parallel < 100000){
            processor.protocol.distributed_comparison_function(processor, instruction, n);
        }
        else{
            processor.protocol.distributed_comparison_function_gpu(processor, instruction, n);
        }
        
    }

    // if (tag == string("LTZ\0", 4))
    // {
    //     octetStream cs;
    //     if(P.my_num() == GEN){  
            
            
    //         this->fss3prep->gen_fake_dcf(1, n);            

    //         signal = 1;
    //         signal.pack(cs); 
    //         P.send_to(EVAL_1, cs);
    //         P.send_to(EVAL_2, cs);
    //     }
    //     // This comparison is designed according to the DRELU protocol in 'Orca: FSS-based Secure Training with GPUs'.
    //     else{
    //         P.receive_player(GEN, cs);
    //         signal.unpack(cs);
    //     }
    //     if(signal){
    //         processor.protocol.distributed_comparison_function(processor, instruction, n);
    //     } 
    // }
}


bigint get_mod(bigint x, int k){
    bigint res;
    res = x - ((x >> k) << k);
    return res;
}

// instructions for rfss3
template<class T>
void Fss<T>::conv2d_relu_rfss3s(SubProcessor<T> &proc, const Instruction& instruction)
{
    std::cout << "executing conv2d_relu_rfss3 in Processor.hpp" << std::endl;

    proc.protocol.init_dotprod_without_trunc();
    auto& args = instruction.get_start();
    int output_h = args[0], output_w = args[1];
    int inputs_h = args[2], inputs_w = args[3];
    int weights_h = args[4], weights_w = args[5];
    int stride_h = args[6], stride_w = args[7];
    int n_channels_in = args[8];
    int padding_h = args[9];
    int padding_w = args[10];
    int batch_size = args[11];
    size_t r0 = instruction.get_r(0);
    size_t r1 = instruction.get_r(1);
    int r2 = instruction.get_r(2);
    int lengths[batch_size][output_h][output_w];
    memset(lengths, 0, sizeof(lengths));
    int filter_stride_h = 1;
    int filter_stride_w = 1;
    if (stride_h < 0)
    {
        filter_stride_h = -stride_h;
        stride_h = 1;
    }
    if (stride_w < 0)
    {
        filter_stride_w = -stride_w;
        stride_w = 1;
    }

    for (int i_batch = 0; i_batch < batch_size; i_batch ++)
    {
        size_t base = r1 + i_batch * inputs_w * inputs_h * n_channels_in;
        assert(base + inputs_w * inputs_h * n_channels_in <= proc.S.size());
        T* input_base = &proc.S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                int in_x_origin = (out_x * stride_w) - padding_w;
                int in_y_origin = (out_y * stride_h) - padding_h;

                for (int filter_y = 0; filter_y < weights_h; filter_y++)
                {
                    int in_y = in_y_origin + filter_y * filter_stride_h;
                    if ((0 <= in_y) and (in_y < inputs_h))
                        for (int filter_x = 0; filter_x < weights_w; filter_x++)
                        {
                            int in_x = in_x_origin + filter_x * filter_stride_w;
                            if ((0 <= in_x) and (in_x < inputs_w))
                            {
                                T* pixel_base = &input_base[(in_y * inputs_w
                                        + in_x) * n_channels_in];
                                T* weight_base = &proc.S[r2
                                        + (filter_y * weights_w + filter_x)
                                                * n_channels_in];
                                for (int in_c = 0; in_c < n_channels_in; in_c++)
                                    proc.protocol.prepare_dotprod_without_trunc(pixel_base[in_c],
                                            weight_base[in_c]);
                                lengths[i_batch][out_y][out_x] += n_channels_in;
                            }
                        }
                }
                proc.protocol.next_dotprod_without_trunc();
            }
    }

    // protocol.exchange();

    for (int i_batch = 0; i_batch < batch_size; i_batch ++)
    {
        size_t base = r0 + i_batch * output_h * output_w;
        assert(base + output_h * output_w <= proc.S.size());
        T* output_base = &proc.S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                output_base[out_y * output_w + out_x] =
                        proc.protocol.finalize_dotprod_without_trunc(lengths[i_batch][out_y][out_x]);
            }
    }

    std::cout << "total size is " << batch_size * output_h * output_w << std::endl;
    int n = args[12] * 2 + args[13] + 1;
    bigint msb_xhat, tmp_select_bit, select_bit[batch_size * output_h * output_w], final_result;
    
    typename T::open_type tmp_bit, tmp_sub = (1LL << (n-1)), masked_x[batch_size * output_h * output_w], eval_value[batch_size * output_h * output_w];
    // relu
    this->init_fss_conv_relu_prep(proc, args[12]);
    // reveal x + r_mask and save it in proc.S[i][0]
    octetStream cs, select_cs;
    if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
        for(int i = 0; i < batch_size * output_h * output_w; i++){
            std::cout << "proc.S[i][0] is " << proc.S[i+r0][0] << std::endl;
            proc.S[i+r0][0] = proc.S[i+r0][0] + this->fss3prep->r_mask_share;
            proc.S[i+r0][0].pack(cs);
        }
        P.send_to((P.my_num()^1), cs);
        P.receive_player((P.my_num()^1), cs);
        for(int i = 0; i < batch_size * output_h * output_w; i++){
            masked_x[i].unpack(cs);
            proc.S[i][0+r0] = proc.S[i][0+r0] + masked_x[i];
            masked_x[i] = proc.S[i][0+r0];
            std::cout << "masked_x is " << masked_x[i] << std::endl;
            msb_xhat = masked_x[i].get_bit(n-1);
            std::cout << "msb x_hat is " << msb_xhat << std::endl;
            if(msb_xhat){
                eval_value[i] = masked_x[i] - ((masked_x[i] >> (n-1)) << (n-1));
            }
            else{
                eval_value[i] = masked_x[i];
            }
            
            eval_value[i] = tmp_sub - eval_value[i] - 1;
            
            select_bit[i] = bigint(typename T::open_type(this->evaluate_conv_relu(eval_value[i], n, 1)));
            // std::cout << "select bit is " << select_bit[i] << "----- is " << bigint(typename T::open_type(select_bit[i])) << " d_relu_share " << this->fss3prep->r_drelu_share << std::endl;
            
            select_bit[i] = (select_bit[i] ^ this->fss3prep->r_drelu_share) ^ (P.my_num() * msb_xhat);
            std::cout << "select result is " << select_bit[i] << std::endl;
            select_bit[i] = select_bit[i].get_ui()%2;
            std::cout << "select result is " << select_bit[i] << " r_drelu is " << this->fss3prep->r_drelu_share << std::endl;
            select_bit[i] = select_bit[i] ^ this->fss3prep->r_select_share;
            
            select_bit[i].pack(select_cs);
        }       
        P.send_to((P.my_num()^1), select_cs);
        P.receive_player((P.my_num()^1), select_cs);
        for(int i = 0; i < batch_size * output_h * output_w; i++){
            tmp_select_bit.unpack(select_cs);
            select_bit[i] = select_bit[i] ^ tmp_select_bit;
            tmp_bit = (masked_x[i] >> (n-1));
            std::cout << tmp_bit << std::endl;
            if(select_bit[i].get_ui()){
                if(tmp_bit.get_bit(0)){
                    std::cout << "--------s=1, t=1-------" << std::endl;
                    // std::cout << "masked_x is " << masked_x[i] << std::endl;
                    // std::cout << "masked_x msb is " << msb_xhat << std::endl;
                    // std::cout << "float_bit is " << args[12] << std::endl;
                    // std::cout << "this->fss3prep->u_select_share is " << this->fss3prep->u_select_share << std::endl;
                    // std::cout << "get_mod((masked_x[i]>>args[12]),n-args[12]-1) is " << get_mod((masked_x[i]>>args[12]),n-args[12]-1) << std::endl;
                    // std::cout << "this->fss3prep->o_select_share is " << this->fss3prep->o_select_share << std::endl;
                    // std::cout << "(this->fss3prep->w_select_share << (n - args[12] - 1)) is " << (this->fss3prep->w_select_share << (n - args[12] - 1)) << std::endl;
                    final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->o_select_share + this->fss3prep->v_select_share + ((this->fss3prep->reverse_1_u_select_share) << (n-args[12]-1));
                }
                else{
                    std::cout << "--------s=1, t=0-------" << std::endl;
                    // std::cout << "masked_x is " << masked_x[i] << std::endl;
                    // std::cout << "masked_x msb is " << msb_xhat << std::endl;
                    // std::cout << "float_bit is " << args[12] << std::endl;
                    // std::cout << "this->fss3prep->u_select_share is " << this->fss3prep->u_select_share << std::endl;
                    // std::cout << "get_mod((masked_x[i]>>args[12]),n-args[12]-1) is " << get_mod((masked_x[i]>>args[12]),n-args[12]-1) << std::endl;
                    // std::cout << "this->fss3prep->v_select_share is " << this->fss3prep->v_select_share << std::endl;
                    // std::cout << "(this->fss3prep->w_select_share << (n - args[12] - 1)) is " << (this->fss3prep->w_select_share << (n - args[12] - 1)) << std::endl;
                    final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1)-this->fss3prep->o_select_share + this->fss3prep->v_select_share + ((this->fss3prep->p_select_share - this->fss3prep->w_select_share) << (n-args[12]-1));
                }
                
            }
            else{
                if(tmp_bit.get_bit(0)){
                    std::cout << "--------s=0, t=1-------" << std::endl;
                    // std::cout << "masked_x is " << masked_x[i] << std::endl;
                    // std::cout << "masked_x msb is " << msb_xhat << std::endl;
                    // std::cout << "float_bit is " << args[12] << std::endl;
                    // std::cout << "this->fss3prep->u_select_share is " << this->fss3prep->u_select_share << std::endl;
                    // std::cout << "get_mod((masked_x[i]>>args[12]),n-args[12]-1) is " << get_mod((masked_x[i]>>args[12]),n-args[12]-1) << std::endl;
                    // std::cout << "this->fss3prep->v_select_share is " << this->fss3prep->v_select_share << std::endl;
                    // std::cout << "(this->fss3prep->w_select_share << (n - args[12] - 1)) is " << (this->fss3prep->w_select_share << (n - args[12] - 1)) << std::endl;
                    final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->v_select_share + (this->fss3prep->reverse_u_select_share << (n-args[12]-1));
                }
                else{
                    std::cout << "-------s=0, t=0-------" << std::endl;
                    // std::cout << "masked_x is " << masked_x[i] << std::endl;
                    // std::cout << "masked_x msb is " << msb_xhat << std::endl;
                    // std::cout << "float_bit is " << args[12] << std::endl;
                    std::cout << "this->fss3prep->u_select_share is " << this->fss3prep->u_select_share << std::endl;
                    // std::cout << "get_mod((masked_x[i]>>args[12]),n-args[12]-1) is " << get_mod((masked_x[i]>>args[12]),n-args[12]-1) << std::endl;
                    // std::cout << "this->fss3prep->v_select_share is " << this->fss3prep->v_select_share << std::endl;
                    // std::cout << "((this->fss3prep->reverse_u_select_share << (n - args[12] - 1)) is " << (this->fss3prep->reverse_u_select_share << (n - args[12] - 1)) << std::endl;
                    final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->v_select_share + (this->fss3prep->w_select_share << (n-args[12]-1));
                }
            }
        }
        std::cout << "final result is " << get_mod(final_result, n) << std::endl;
        return;
    }

}

#endif // PROTOCOLS_FSS_HPP