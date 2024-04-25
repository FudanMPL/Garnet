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
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
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
        this->fss3prep->init_offline_values(&processor, 0);
    }
    
    return;
}

template<class T>
void Fss<T>::init_fss_conv_relu_prep(SubProcessor<T> &processor, int float_bits, int case_num=1){
    bigint signal = 0;
    octetStream cs;
    if(this->fss3prep == nullptr){
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
        // 有些情况只需要读取不需要更新
        if(case_num){
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
        }
        this->fss3prep->init_offline_values(&processor, 1);
    }
    return;
}

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

template<class T>
void Fss<T>::distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    auto& args = instruction.get_start();
    int dcf_parallel = int(args[2]), output_base = args[0], input_base = args[1];
    typename T::open_type result[dcf_parallel];
    bigint res_bool[dcf_parallel], tmp_bool_res, tmp_val;
    fstream r;
    octetStream cs[2], reshare_cs, b_cs; //cs0, cs1; 
    MC->init_open(P, n);
    for(size_t i = 0; i < dcf_parallel; i+= 1){ 
        // reveal x + r
        auto dest = &proc.S[input_base + i][0];
        *dest = *dest + typename T::open_type(this->fss3prep->r_share);
        MC->prepare_open(proc.S[input_base + i]);   
    }
    MC->exchange(P);

    if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
        bigint msb_xhat,  tb[dcf_parallel], res;
        typename T::open_type tmp[dcf_parallel], tmp_sub = (1LL << (n-1));
        for(size_t i = 0; i < dcf_parallel; i+= 1){ 
            result[i] = MC->finalize_raw();
            msb_xhat = result[i].get_bit(n - 1);
            if(msb_xhat){
                result [i] = result[i] - (1LL << (n-1));
            }
            result[i] = tmp_sub - result[i] - 1;
            tb[i] = this->evaluate(result[i], n, 1, 0);
            tb[i] = bigint(typename T::open_type(tb[i])) ^ bigint(typename T::open_type(this->fss3prep->r_b)) ^ (P.my_num() * msb_xhat);
            res_bool[i] = bigint(typename T::open_type(tb[i])).get_ui() % 2;
            res_bool[i] = res_bool[i] ^ this->fss3prep->rs_b;
            res_bool[i].pack(b_cs);
        }
        P.send_to((P.my_num()^1), b_cs);
        P.receive_player((P.my_num()^1), b_cs);
        for(size_t i = 0; i < dcf_parallel; i+= 1){   
            tmp_bool_res.unpack(b_cs);
            res_bool[i] = res_bool[i] ^ tmp_bool_res;
            if(res_bool[i])
                res = P.my_num()-this->fss3prep->u-this->fss3prep->r_in_2+this->fss3prep->w;
            else
                res = this->fss3prep->u +  this->fss3prep->w - this->fss3prep->z;
            proc.S[output_base + i][P.my_num()^1] = this->fss3prep->rr_share;
            this->fss3prep->rr_share_tmp = bigint(typename T::open_type(P.my_num()-res)) - this->fss3prep->rr_share;
            proc.S[output_base + i][P.my_num()] = this->fss3prep->rr_share_tmp;
            this->fss3prep->rr_share_tmp.pack(reshare_cs);
        }
        P.send_to((P.my_num()^1), reshare_cs);
        P.receive_player((P.my_num()^1), reshare_cs);
        for(size_t i = 0; i < dcf_parallel; i+= 1){ 
            this->fss3prep->rr_share_tmp.unpack(reshare_cs);
            proc.S[output_base + i][P.my_num()] = proc.S[output_base + i][P.my_num()] + this->fss3prep->rr_share_tmp;   
        }
    }
    else{ 
        for(size_t i = 0; i < dcf_parallel; i+=1){ 
            {
                proc.S[output_base + i][0] = this->fss3prep->rr_share;
                proc.S[output_base + i][1] = this->fss3prep->rr_share_tmp;         
            }
        }
        return;
    }
}

template<class T>
void Fss<T>::distributed_comparison_function_gpu(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
    
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
        
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->tcw[j][i-1]);
        }

        if(n <= 128){
            convert[0] = v_hat[0] >> (lambda-result_length);
            convert[1] = v_hat[1] >> (lambda-result_length);
        }

        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]) + (1^b) * (convert[xi] + tmp_t * this->fss3prep->vcw[i-1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
    }

    if(n <= 128)
        convert[0] = s[xi] >> (lambda-result_length);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * this->fss3prep->final_cw) + (1^b) * (convert[0] + tmp_t * this->fss3prep->final_cw);
    return tmp_v;  
}

template<class T>
bigint Fss<T>::evaluate_conv_relu(typename T::open_type x, int n, int result_length){
    PRNG prng;
    int b = P.my_num(), xi;
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda = 127, lambda_bytes = 16;
    
    octet seed[lambda_bytes];
    // r is the random value generate by GEN
    bigint s_hat[2], v_hat[2], s[2], v[2], convert[2], tmp_bigint, tmp_v, tmp_out;
    bool t_hat[2], t[2], tmp_t;
    bytesFromBigint(&seed[0], this->fss3prep->full_seed_bigint, lambda_bytes);
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
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->full_tcw[j][i-1]);
        }

        // std::cout << "v_hat[0], v_hat[1] are " << v_hat[0] << " " << v_hat[1] << std::endl;
        if(n <= 128){
            // std::cout << "length is " << lambda-n << std::endl;
            convert[0] = v_hat[0] >> (lambda-result_length);
            convert[1] = v_hat[1] >> (lambda-result_length);
        }

        // std::cout << "convert[0], convert[1] are " << convert[0] << " " << convert[1] << std::endl;
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * this->fss3prep->full_vcw[i-1]) + (1^b) * (convert[xi] + tmp_t * this->fss3prep->full_vcw[i-1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
        // std::cout << tmp_v << std::endl;
    }

    if(n <= 128)
        convert[0] = s[xi] >> (lambda-result_length);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * this->fss3prep->full_final_cw) + (1^b) * (convert[0] + tmp_t * this->fss3prep->full_final_cw);
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
    add_shares.push_back(dotprod_share);
    dotprod_share.assign_zero();
}

template <class T>
inline T Fss<T>::finalize_dotprod_without_trunc(int length)
{
    T result;
    result[0] = add_shares.next();
    result[1] = 0;
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
void Fss<T>::fss_cmp(SubProcessor<T> &processor, const Instruction &instruction){
    auto& args = instruction.get_start();
    int n = args[3] + args[4] * 2;
    this->init_fss_cmp_prep(processor);
    processor.protocol.distributed_comparison_function(processor, instruction, n);
}



bigint get_mod(bigint x, int k){
    bigint res;
    res = x - ((x >> k) << k);
    return res;
}

void bit_transfer(bigint &res, bigint tmp, int cnt){
    res = res + (tmp << cnt);
}

// instructions for rfss3
template<class T>
void Fss<T>::conv2d_rfss3s(SubProcessor<T> &proc, const Instruction& instruction)
{
    auto& args = instruction.get_start();
    proc.protocol.init_dotprod_without_trunc();
    proc.protocol.init_fss_conv_relu_prep(proc, args[12]);
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

    
}

template<class T>
void Fss<T>::trunc_relu_rfss3s(SubProcessor<T> &proc, const Instruction& instruction){
    // auto& args = instruction.get_start();
    // proc.protocol.init_fss_conv_relu_prep(proc, args[12]);
    // int n = args[12] * 2 + args[13] + 1;
    // bigint msb_xhat, tmp_select_bit, select_bit[batch_size * output_h * output_w], final_result, tmp_reshare;
    // typename T::open_type tmp_bit, tmp_sub = (1LL << (n-1)), masked_x[batch_size * output_h * output_w], eval_value[batch_size * output_h * output_w];
    // // relu
    // octetStream cs, select_cs, reshare_cs;
    // if(P.my_num() == EVAL_1 || P.my_num() == EVAL_2){
    //     for(int i = 0; i < batch_size * output_h * output_w; i++){
    //         proc.S[i+r0][0] = proc.S[i+r0][0] + this->fss3prep->r_mask_share;
    //         proc.S[i+r0][0].pack(cs);
    //     }
    //     P.send_to((P.my_num()^1), cs);
    //     P.receive_player((P.my_num()^1), cs);
    //     for(int i = 0; i < batch_size * output_h * output_w; i++){
    //         masked_x[i].unpack(cs);
    //         proc.S[i+r0][0] = proc.S[i+r0][0] + masked_x[i];
    //         masked_x[i] = proc.S[i+r0][0];
    //         msb_xhat = masked_x[i].get_bit(n-1);
    //         if(msb_xhat){
    //             eval_value[i] = masked_x[i] - ((masked_x[i] >> (n-1)) << (n-1));
    //         }
    //         else{
    //             eval_value[i] = masked_x[i];
    //         }
    //         eval_value[i] = tmp_sub - eval_value[i] - 1;
    //         select_bit[i] = bigint(typename T::open_type(this->evaluate_conv_relu(eval_value[i], n, 1)));
    //         select_bit[i] = (select_bit[i] ^ this->fss3prep->r_drelu_share) ^ (P.my_num() * msb_xhat);
    //         select_bit[i] = select_bit[i].get_ui()%2;
    //         select_bit[i] = select_bit[i] ^ this->fss3prep->r_select_share;
    //         bit_transfer(tmp_select_bit, select_bit[i], i);
    //         select_bit[i].pack(select_cs);
    //     }       
    //     std::cout << "tmp_select_bit is " << tmp_select_bit << std::endl;
    //     P.send_to((P.my_num()^1), select_cs);
    //     P.receive_player((P.my_num()^1), select_cs);
    //     for(int i = 0; i < batch_size * output_h * output_w; i++){
    //         tmp_select_bit.unpack(select_cs);
    //         select_bit[i] = select_bit[i] ^ tmp_select_bit;
    //         tmp_bit = (masked_x[i] >> (n-1));
    //         if(select_bit[i].get_ui()){
    //             if(tmp_bit.get_bit(0)){
    //                 final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->o_select_share + this->fss3prep->v_select_share + ((this->fss3prep->reverse_1_u_select_share) << (n-args[12]-1));
    //             }
    //             else{
    //                 final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1)-this->fss3prep->o_select_share + this->fss3prep->v_select_share + ((this->fss3prep->p_select_share - this->fss3prep->w_select_share) << (n-args[12]-1));
    //             }
    //         }
    //         else{
    //             if(tmp_bit.get_bit(0)){
    //                 final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->v_select_share + (this->fss3prep->reverse_u_select_share << (n-args[12]-1));
    //             }
    //             else{
    //                 final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i],n)>>args[12]),n-args[12]-1) - this->fss3prep->v_select_share + (this->fss3prep->w_select_share << (n-args[12]-1));
    //             }
    //         }
    //         proc.S[r0+i][(P.my_num())^1] = this->fss3prep->reshare_value;
    //         tmp_reshare = final_result - this->fss3prep->reshare_value;
    //         proc.S[r0+i][(P.my_num())] = tmp_reshare;
    //         tmp_reshare.pack(reshare_cs);
    //     }
    //     P.send_to((P.my_num()^1), reshare_cs);
    //     P.receive_player((P.my_num()^1), reshare_cs);
    //     for(int i = 0; i < batch_size * output_h * output_w; i++){
    //         this->fss3prep->reshare_value_tmp.unpack(reshare_cs);
    //         proc.S[r0+i][(P.my_num())] = proc.S[r0+i][(P.my_num())] + this->fss3prep->reshare_value_tmp;
    //     }
    // }
    // else{
    //     for(int i = 0; i < batch_size * output_h * output_w; i++){
    //         proc.S[r0+i][0] = this->fss3prep->reshare_value;
    //         proc.S[r0+i][1] = this->fss3prep->reshare_value_tmp;
    //     }
    // }
    return;
}
#endif // PROTOCOLS_FSS_HPP