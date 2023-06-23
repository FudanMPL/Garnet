/*
 * FSS.hpp
 *
 */

#ifndef PROTOCOLS_FSS_HPP_
#define PROTOCOLS_FSS_HPP_

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

//-------------------
#include <math.h>

template<class T>
void Fss<T>::init(Preprocessing<T>& prep, typename T::MAC_Check& MC)
{
    this->prep = &prep;
    this->MC = &MC;
}




template <class T>
void Fss<T>::distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int lambda)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    typename T::clear result, r_tmp, dcf_u, dcf_v;
    auto& args = instruction.get_start();
    for(size_t i = 0; i < args.size(); i+= args[i]){ 
        fstream r;
        octetStream cs[2], reshare_cs, t_cs; //cs0, cs1; 
        
        r.open("Player-Data/2-fss/r" + to_string(P.my_num()), ios::in);
        r >> r_tmp;
        r.close();
        MC->init_open(P, lambda);
        auto dest = &proc.S[args[i+3]][0];
        *dest = *dest + r_tmp;
        MC->prepare_open(proc.S[args[i+3]]);
        MC->exchange(P);
        // std::cout << "Initial x " << proc.S[args[i+3]][0] << std::endl;
        result = MC->finalize_raw();
        if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2)
        {
            bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
            dcf_res_u = this->evaluate(result, lambda);
            result += 1LL<<(lambda-1);
            dcf_res_v = this->evaluate(result, lambda);
        
            auto size = dcf_res_u.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
            size = dcf_res_v.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));

            if(result.get_bit(lambda)){
                // dcf_res = dcf_res_v - dcf_res_u + P.my_num();
                r_tmp = dcf_v - dcf_u + P.my_num();
            }
            else{
                // dcf_res = dcf_res_v - dcf_res_u;
                r_tmp = dcf_v - dcf_u;
            }

            // Test for reveal
            // octetStream cs_reveal_u, cs_reveal_v, cs_reveal;
            // dcf_res_u.pack(cs_reveal_u);
            // P.send_to(1^P.my_num(), cs_reveal_u);
            // P.receive_player(1^P.my_num(), cs_reveal_u);
            // bigint rec;
            // rec.unpack(cs_reveal_u);
            // dcf_res_u += rec;

            // dcf_res_v.pack(cs_reveal_v);
            // P.send_to(1^P.my_num(), cs_reveal_v);
            // P.receive_player(1^P.my_num(), cs_reveal_v);
            // rec.unpack(cs_reveal_v);
            // dcf_res_v += rec;

            // dcf_res.pack(cs_reveal);
            // P.send_to(1^P.my_num(), cs_reveal);
            // P.receive_player(1^P.my_num(), cs_reveal);
            // rec.unpack(cs_reveal);
            // dcf_res_reveal = dcf_res + rec;
            // std::cout << "R_tmp is " << r_tmp << " Initial result is " << result - 1LL<<(lambda-1) << " Result is " << result 
            // << " Its signal bit is " << result.get_bit(lambda-1) << " Reveal value of dcf_u is : " << dcf_res_u << " dcf_v is: " 
            // << dcf_res_v << " dcf_res " << dcf_res_reveal << std::endl;

            // std::cout << "---------------" << std::endl;
            // std::cout << "Bit decomposition of result is :" << std::endl;
            // for(int i = lambda - 1; i >= 0; i--){
            //     std::cout << result.get_bit(i);
            // }
            // std::cout << std::endl;

            // r_tmp = 0;
            // auto size = dcf_res.get_mpz_t()->_mp_size;
            // mpn_copyi((mp_limb_t*)r_tmp.get_ptr(), dcf_res.get_mpz_t()->_mp_d, abs(size));

            // r_tmp = 0;
            // for(int i = ceil(lambda/64.0) - 1; i >= 0 ; i--){
            //     r_tmp = r_tmp << 64;
            //     std::cout << i<<"-th block is " << dcf_res.get_mpz_t()->_mp_d[i] << std::endl;
            //     typename T::clear dcf_inter = dcf_res.get_mpz_t()->_mp_d[i];
            //     r_tmp = r_tmp + dcf_inter;
            // }
            // std::cout << "r_tmp is " << r_tmp << std::endl;
            if(size < 0)
                r_tmp = - r_tmp;
            P.receive_player(GEN, cs[P.my_num()]);
            typename T::clear tmp;
            tmp.unpack(cs[P.my_num()]);
            proc.S[args[i+2]][0] = typename T::clear(P.my_num()) - r_tmp - tmp;
        }
        else{
            typename T::clear r_sum, r0 = 0, r1 = 0, r_inter = 0;
            bigint r_tmp;
            prng.get(r_tmp, lambda); 
            auto size = r_tmp.get_mpz_t()->_mp_size;
            for(int i = ceil(lambda/64.0) - 1; i >= 0 ; i--){
                r0 = r0 << 64;
                r_inter = r_tmp.get_mpz_t()->_mp_d[i];
                r0 = r0 + r_inter;
            }
            if(size < 0)
                r0 = - r0;


            prng.get(r_tmp, lambda); 
            size = r_tmp.get_mpz_t()->_mp_size;

            for(int i = ceil(lambda/64.0) - 1; i >= 0 ; i--){
                r1 = r1 << 64;
                r_inter = r_tmp.get_mpz_t()->_mp_d[i];
                r1 = r1 + r_inter;
            }
            if(size < 0)
                r0 = - r0;
            r_sum = r0 + r1;
            proc.S[args[i+2]][0] = r_sum;
            r0.pack(cs[EVAL_1]);
            P.send_to(EVAL_1, cs[EVAL_1]);
            r1.pack(cs[EVAL_2]);
            P.send_to(EVAL_2, cs[EVAL_2]);
        }
        proc.S[args[i+2]][0].pack(reshare_cs);
        P.send_to((P.my_num()+1)%P.num_players(), reshare_cs);
        P.receive_player((P.my_num()+2)%P.num_players(), reshare_cs);
        proc.S[args[i+2]][1].unpack(reshare_cs);   
    }
    return;
}

template <class T>
void Fss<T>::generate(){
    //判断buffer里面有没有数
    //有数值就读出即可
    //没有数就gen_dcp
}


template<class T>
bigint Fss<T>::evaluate(typename T::clear x, int lambda){
    fstream k_in;
    PRNG prng;
    prng.ReSeed();
    int b = P.my_num(), xi;
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = max(16, (lambda)/8);
    k_in.open("Player-Data/2-fss/k" + to_string(P.my_num()), ios::in);
    octet seed[lambda_bytes], tmp_seed[lambda_bytes];
    // r is the random value generate by GEN
    bigint s_hat[2], v_hat[2], t_hat[2], s[2], v[2], t[2], scw, vcw, tcw[2], convert[2], cw, tmp_t, tmp_v, tmp_out;
    k_in >> tmp_t;
    bytesFromBigint(&seed[0], tmp_t, lambda_bytes);
    // std::cout << "init seed is " << tmp_t << std::endl;
    tmp_t = b;
    tmp_v = 0;
    for(int i = 0; i < lambda - 1; i++){
        xi = x.get_bit(lambda - i - 1);
        bigintFromBytes(tmp_out, &seed[0], 16);
        k_in >> scw >> vcw >> tcw[0] >> tcw[1];
        prng.SetSeed(seed);
        for(int j = 0; j < 2; j++){
            prng.get(t_hat[j], 1);
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j] ,lambda);
            s[j] = s_hat[j] ^ (tmp_t * scw);
            t[j] = t_hat[j] ^ (tmp_t * tcw[j]);
        }  
        bytesFromBigint(&tmp_seed[0], v_hat[0], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[0], lambda); 
        bytesFromBigint(&tmp_seed[0], v_hat[1], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[1], lambda);
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * vcw) + (1^b) * (convert[xi] + tmp_t * vcw);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
    }
    k_in >> cw;
    k_in.close();
    prng.SetSeed(seed);
    prng.get(convert[0], lambda);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * cw) + (1^b) * (convert[0] + tmp_t * cw);
    return tmp_v;  
}

template <class T>
Fss<T>::Fss(Player &P) : ReplicatedBase(P)
{
    assert(T::vector_length == 2);
}

template <class T>
Fss<T>::Fss(const ReplicatedBase &other) : ReplicatedBase(other)
{
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
                S[info.dest_base + i] += ((r_msb + c_dprime)
                                          << (info.k - info.m));
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
    // auto& args = instruction.get_start();
    octetStream cs;
    int r0 = instruction.get_r(0), lambda = T::clear::MAX_N_BITS;
    // std::cout << "Bit length is " << lambda << std::endl;
    bigint signal = 0;
    string tag((char *)&r0, 4);
    // std::cout << tag << std::endl;
    if (tag == string("LTZ\0", 4))
    {
        if(P.my_num() == GEN){  
            this->fss3prep->gen_fake_dcp(1, lambda);  
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        // This comparison is designed according to the DRELU protocol in 'Orca: FSS-based Secure Training with GPUs'.
        else{
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        if(signal){
            processor.protocol.distributed_comparison_function(processor, instruction, lambda);
        }
    }
}

#endif // PROTOCOLS_FSS_HPP