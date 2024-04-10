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
void Fss<T>::init_fss_prep(SubProcessor<T> &processor){
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
            tb[cnt] = this->evaluate(result[cnt], n);
            tb[cnt] = bigint(typename T::open_type(tb[cnt])) ^ bigint(typename T::open_type(this->fss3prep->r_b)) ^ (P.my_num() * msb_xhat);
            res_bool[cnt] = bigint(typename T::open_type(tb[cnt])).get_ui() % 2;
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
                res = P.my_num() -  this->fss3prep->u -  this->fss3prep->r_in_2 +  this->fss3prep->w;
            else
                res =  this->fss3prep->u +  this->fss3prep->w - this->fss3prep->z;
            // std::cout << bigint(typename T::open_type(res)) << std::endl;
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
bigint Fss<T>::evaluate(typename T::open_type x, int n){
    
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
    
    for(int i = 1; i < n-16 ; i++){
        xi = x.get_bit(n - i - 1);
        
        prng.SetSeed(seed);
        for(int j = 0; j < 2; j++){
            // prng.get(tmp_out, 2*lambda+2);
            // mpn_copyi(tmp_out.get_mpz_t()->_mp_d, s_hat[j].get_mpz_t()->_mp_d, s_hat[j].get_mpz_t()->_mp_size);
            // mpn_copyi(tmp_out.get_mpz_t()->_mp_d + s_hat[j].get_mpz_t()->_mp_size, v_hat[j].get_mpz_t()->_mp_d, v_hat[j].get_mpz_t()->_mp_size);
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j], lambda);
            t_hat[j] = s_hat[j].get_ui() & 1;
            s[j] = s_hat[j] ^ (tmp_t * this->fss3prep->scw[i-1]);
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->tcw[j][i-1]);
        }

        // std::cout << "v_hat[0], v_hat[1] are " << v_hat[0] << " " << v_hat[1] << std::endl;
        if(n <= 128){
            // std::cout << "length is " << lambda-n << std::endl;
            convert[0] = v_hat[0] >> (lambda-n);
            convert[1] = v_hat[1] >> (lambda-n);
        }

        // std::cout << "convert[0], convert[1] are " << convert[0] << " " << convert[1] << std::endl;
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]) + (1^b) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
        // std::cout << tmp_v << std::endl;
    }

    if(n <= 128)
        convert[0] = s[xi] >> (lambda-n);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * this->fss3prep->final_cw) + (1^b) * (convert[0] + tmp_t * this->fss3prep->final_cw);
    // std::cout << "final_tmp_v" << tmp_v << std::endl;
    return tmp_v;  
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
        this->init_fss_prep(processor);
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

#endif // PROTOCOLS_FSS_HPP