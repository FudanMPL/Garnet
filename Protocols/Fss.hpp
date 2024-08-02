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
void Fss<T>::Muliti_Interval_Containment(SubProcessor<T> &proc, const Instruction &instruction, int lambda){
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    // auto& args = instruction.get_start();
    // int res_address = args[2], x_base = args[3], spline_base = args[4], length = args[5];
    // typename T::clear tmp, r_tmp, eval_res[length+1], spline_res[length],z; 
    // vector<typename T::clear> x_add_r;
    // bigint dcf[length];
    // fstream r;
    // r.open("Player-Data/2-fss/multi_r" + to_string(P.my_num()), ios::in);
    // r >> r_tmp;
    // // for(int i = 0; i < args.size(); i+=args[0]){

    // // }
    // for(int i = 0; i < args.size(); i+=args[0]){
    //     fstream r;
    //     octetStream cs[2], reshare_cs, t_cs; //cs0, cs1; 
        
    //     r.open("Player-Data/2-fss/multi_r" + to_string(P.my_num()), ios::in);
    //     r >> r_tmp;
    //     MC->init_open(P, lambda);
    //     auto dest = &proc.S[x_base + i][0];
    //     *dest = *dest + r_tmp;
    //     MC->prepare_open(proc.S[args[x_base + i]]);
    //     MC->exchange(P);
    //     x_add_r = MC->finalize_raw();
    //     std::cout << "x+r is " << x_add_r << std::endl;

    //     //res0 = 0，后面才存放Eval结果
    //     if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
    //         auto size = dcf[0].get_mpz_t()->_mp_size;
    //         eval_res[0] = 0;
    //         r >> z;
    //         r_tmp = 0;
    //         for(int j = 0; j < length; j++){
    //             tmp = x_add_r - proc.C[spline_base + j];
    //             dcf[j] = this->evaluate(tmp, lambda);
    //             size = dcf[j].get_mpz_t()->_mp_size;
    //             mpn_copyi((mp_limb_t*)eval_res[j+1].get_ptr(), dcf[j].get_mpz_t()->_mp_d, abs(size));

    //             // evaluation
    //             if(size < 0){
    //                 eval_res[j+1] = -eval_res[j+1];
    //             }
    //             if(lambda == 128){
    //                 if(j == 0){
    //                     spline_res[j] = typename T::clear(P.my_num()) * (1  - (x_add_r - proc.C[spline_base+j]).get_bit(lambda)) - eval_res[j] + eval_res[j+1] + z;
    //                 }
    //                 else{
    //                     spline_res[j] = typename T::clear(P.my_num()) * ((x_add_r - proc.C[spline_base+j-1]).get_bit(lambda)  - (x_add_r - proc.C[spline_base+j]).get_bit(lambda)) - eval_res[j] + eval_res[j+1] + z;
    //                 }
    //             }
    //             else{
    //                 if(j == 0){
    //                     spline_res[j] = typename T::clear(P.my_num()) * (1  - (x_add_r - proc.C[spline_base+j]).get_bit(lambda - 1)) - eval_res[j] + eval_res[j+1] + z;
    //                 }
    //                 else{
    //                     spline_res[j] = typename T::clear(P.my_num()) * ((x_add_r - proc.C[spline_base+j-1]).get_bit(lambda - 1)  - (x_add_r - proc.C[spline_base+j]).get_bit(lambda - 1)) - eval_res[j] + eval_res[j+1] + z;
    //                 }
    //             }
    //         } 

    //         typename T::clear random_val[length+1];
    //         P.receive_player(GEN, cs[P.my_num()]);
    //         std::cout << "recieving! " << std::endl;
    //         for(int j = 0; j < length+1; j++){
    //             random_val[j].unpack(cs[P.my_num()]);
    //             proc.S[res_address + j][0] = typename T::clear(P.my_num()) + spline_res[j] - random_val[j];
    //         }
        

    //         r.close();
    //     }



    //     else{
    //         typename T::clear r_sum[length+1], r0[length+1], r1[length+1];
    //         for(int j = 0; j < length+1; j++){
    //             bigint r_tmp;
    //             prng.get(r_tmp, lambda); 
    //             auto size = r_tmp.get_mpz_t()->_mp_size;
    //             mpn_copyi((mp_limb_t*)r0[j].get_ptr(), r_tmp.get_mpz_t()->_mp_d, abs(size));
    //             prng.get(r_tmp, lambda); 
    //             size = r_tmp.get_mpz_t()->_mp_size;
    //             mpn_copyi((mp_limb_t*)r1[j].get_ptr(), r_tmp.get_mpz_t()->_mp_d, abs(size));
                
    //             r_sum[j] = r0[j] + r1[j];
    //             proc.S[res_address + j][0] = r_sum[j];
    //             r0[j].pack(cs[EVAL_1]);

    //             std::cout << "packed random value r0 is " << r0[j] << std::endl;
    //             r1[j].pack(cs[EVAL_2]);
    //         }
    //         P.send_to(EVAL_1, cs[EVAL_1]);
    //         P.send_to(EVAL_2, cs[EVAL_2]);
    //         std::cout << "finishing sending!" << std::endl;
    //     }
    // }
    // typename T::clear reshare[length+1], recv_reshare[length+1];
    // for(int j = 0; j < length+1; j++){
    //     reshare[j] = proc.S[res_address + j][0];
    //     reshare[j].pack(reshare_cs);
    // }
    

    // P.send_to((P.my_num()+1)%P.num_players(), reshare_cs);
    // P.receive_player((P.my_num()+2)%P.num_players(), reshare_cs);
    // for(int j = 0; j < length+1; j++){
    //     recv_reshare[j].unpack(reshare_cs);
    //     proc.S[res_address + j][1] = recv_reshare[j];
    // }
}



template <class T>
void Fss<T>::distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int lambda)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);    
    typename T::clear result, r_tmp, dcf_u, dcf_v; 
    auto& args = instruction.get_start();
    fstream r;
    octetStream cs[2], reshare_cs, t_cs; //cs0, cs1; 
    r.open("Player-Data/2-fss/r" + to_string(P.my_num()), ios::in);
    r >> r_tmp;
    r.close();
    MC->init_open(P, lambda);
    for(size_t i = 0; i < args.size(); i+= args[i]){ 
        auto dest = &proc.S[args[i+3]][0];
        *dest = *dest + r_tmp;
        MC->prepare_open(proc.S[args[i+3]]);   
        if(P.my_num() == GEN){
            typename T::clear r_sum, r0 = 0, r1 = 0;
            bigint r_tmp;
            prng.get(r_tmp, lambda); 
            auto size = r_tmp.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)r0.get_ptr(), r_tmp.get_mpz_t()->_mp_d, abs(size));
            prng.get(r_tmp, lambda); 
            size = r_tmp.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)r1.get_ptr(), r_tmp.get_mpz_t()->_mp_d, abs(size));
            if(size < 0)
                r0 = - r0;
            r_sum = r0 + r1;
            proc.S[args[i+2]][0] = r_sum;
            r0.pack(cs[EVAL_1]);
            r1.pack(cs[EVAL_2]);
        }    
    }
    if(P.my_num() == GEN){
        P.send_to(EVAL_1, cs[EVAL_1]);
        P.send_to(EVAL_2, cs[EVAL_2]);
    }
    else{
        P.receive_player(GEN, cs[P.my_num()]);
    }
    MC->exchange(P);
    for(size_t i = 0; i < args.size(); i+= args[i]){ 
        result = MC->finalize_raw();
        if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2)
        {
            bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
            dcf_res_u = this->evaluate(result, lambda);
            result += 1LL<<(lambda-1);
            dcf_res_v = this->evaluate(result, lambda); 
            auto size = dcf_res_u.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
            if(size < 0)
                dcf_u = -dcf_u;
            size = dcf_res_v.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
            if(size < 0)
                dcf_v = -dcf_v;
            if(result.get_bit(lambda)){
                r_tmp = dcf_v - dcf_u + P.my_num();
            }
            else{
                r_tmp = dcf_v - dcf_u;
            }
            if(lambda == 128){
                if(result.get_bit(lambda)){
                    r_tmp = dcf_v - dcf_u + P.my_num();
                }
                else{
                    r_tmp = dcf_v - dcf_u;
                }
            }
            else{
                if(result.get_bit(lambda-1)){
                    r_tmp = dcf_v - dcf_u + P.my_num();
                }
                else{
                    r_tmp = dcf_v - dcf_u;
                }
            }  
            typename T::clear tmp;
            tmp.unpack(cs[P.my_num()]);
            proc.S[args[i+2]][0] = typename T::clear(P.my_num()) - r_tmp - tmp;
        }
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
    int lambda_bytes = 16;
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
        bigintFromBytes(tmp_out, &seed[0], lambda_bytes);
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
    
    int r0 = instruction.get_r(0), lambda = T::clear::MAX_N_BITS;
    bigint signal = 0;
    string tag((char *)&r0, 4);
    // std::cout << tag << std::endl;

    // std::cout << "-----------------------" << std::endl; 
    if (tag == string("LTZ\0", 4))
    {
        octetStream cs;
        if(P.my_num() == GEN){  
            std::cout << "Generating fake dcf " << std::endl;
            this->fss3prep->gen_fake_dcf(1, lambda);  
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
    else{
        auto& args = instruction.get_start();
        // std::cout << "CISC CALLED!" << std::endl;
        std::cout << "Arguments are: " << std::endl;
        for(auto i : args)
            std::cout << i << std::endl;
        octetStream cs;
        std::cout << "MTS CALLED!" << std::endl;
        if(P.my_num() == GEN){  
            std::cout << "Generating fake multi dcf " << std::endl;
            this->fss3prep->gen_fake_multi_spline_dcf(processor, 1, lambda, args[4], args[8]);  
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        else{
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        if(signal){
            processor.protocol.Muliti_Interval_Containment(processor, instruction, lambda);
        }
    }
}

#endif // PROTOCOLS_FSS_HPP