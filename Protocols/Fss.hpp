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
#include "Processor/Conv2dTuple.h"
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
void Fss<T>::init_fss_cmp_prep(SubProcessor<T> &processor)
{
    bigint signal = 0;
    octetStream cs;
    if (this->fss3prep == nullptr)
    {
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
        if (P.my_num() == GEN)
        {
            this->fss3prep->gen_fake_dcf(1, T::clear::MAX_N_BITS);
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        else
        {
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        this->fss3prep->init_offline_values(&processor, 0);
    }

    return;
}

template <class T>
void Fss<T>::init_fss_conv_relu_prep(SubProcessor<T> &processor, int float_bits)
{
    bigint signal = 0;
    octetStream cs;
    if (this->fss3prep == nullptr)
    {
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&processor, usage);
        if (P.my_num() == GEN)
        {
            this->fss3prep->gen_fake_conv_relu(1, T::clear::MAX_N_BITS, float_bits);
            signal = 1;
            signal.pack(cs);
            P.send_to(EVAL_1, cs);
            P.send_to(EVAL_2, cs);
        }
        else
        {
            P.receive_player(GEN, cs);
            signal.unpack(cs);
        }
        this->fss3prep->init_offline_values(&processor, 1);
    }
    return;
}

template <class T>
void Fss<T>::init(Preprocessing<T> &prep, typename T::MAC_Check &MC)
{
    this->prep = &prep;
    this->MC = &MC;
}

template <class T>
void Fss<T>::Muliti_Interval_Containment(SubProcessor<T> &proc, const Instruction &instruction, int lambda)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);

}

template <class T>
void Fss<T>::distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
    PRNG prng;
    prng.ReSeed();
    init(proc.DataF, proc.MC);
    auto &args = instruction.get_start();
    int dcf_parallel = int(args[2]), output_base = args[0], input_base = args[1];
    typename T::open_type result[dcf_parallel];
    bigint res_bool[dcf_parallel], tmp_bool_res, tmp_val;
    fstream r;
    octetStream cs[2], reshare_cs, b_cs; // cs0, cs1;
    MC->init_open(P, n);
    for (size_t i = 0; i < dcf_parallel; i += 1)
    {
        // reveal x+r
        auto dest = &proc.S[input_base+i][0];
        *dest = *dest+typename T::open_type(this->fss3prep->r_share);
        MC->prepare_open(proc.S[input_base+i]);
    }
    MC->exchange(P);

    if (P.my_num() == EVAL_1 || P.my_num() == EVAL_2)
    {
        bigint msb_xhat, tb[dcf_parallel], res;
        typename T::open_type tmp[dcf_parallel], tmp_sub = (1LL << (n - 1));
        for (size_t i = 0; i < dcf_parallel; i += 1)
        {
            result[i] = MC->finalize_raw();
            msb_xhat = result[i].get_bit(n - 1);
            if (msb_xhat)
            {
                result[i] = result[i] - (1LL << (n - 1));
            }
            result[i] = tmp_sub - result[i] - 1;
            tb[i] = this->evaluate(result[i], n, 1, 0);
            tb[i] = bigint(typename T::open_type(tb[i])) ^ bigint(typename T::open_type(this->fss3prep->r_b)) ^ (P.my_num() * msb_xhat);
            res_bool[i] = bigint(typename T::open_type(tb[i])).get_ui() % 2;
            res_bool[i] = res_bool[i] ^ this->fss3prep->rs_b;
            res_bool[i].pack(b_cs);
        }
        P.send_to((P.my_num() ^ 1), b_cs);
        P.receive_player((P.my_num() ^ 1), b_cs);
        for (size_t i = 0; i < dcf_parallel; i += 1)
        {
            tmp_bool_res.unpack(b_cs);
            res_bool[i] = res_bool[i] ^ tmp_bool_res;
            if (res_bool[i])
                res = P.my_num() - this->fss3prep->u - this->fss3prep->r_in_2+this->fss3prep->w;
            else
                res = this->fss3prep->u+this->fss3prep->w - this->fss3prep->z;
            proc.S[output_base+i][P.my_num() ^ 1] = this->fss3prep->rr_share;
            this->fss3prep->rr_share_tmp = bigint(typename T::open_type(P.my_num() - res)) - this->fss3prep->rr_share;
            proc.S[output_base+i][P.my_num()] = this->fss3prep->rr_share_tmp;
            this->fss3prep->rr_share_tmp.pack(reshare_cs);
        }
        P.send_to((P.my_num() ^ 1), reshare_cs);
        P.receive_player((P.my_num() ^ 1), reshare_cs);
        for (size_t i = 0; i < dcf_parallel; i += 1)
        {
            this->fss3prep->rr_share_tmp.unpack(reshare_cs);
            proc.S[output_base+i][P.my_num()] = proc.S[output_base+i][P.my_num()]+this->fss3prep->rr_share_tmp;
        }
    }
    else
    {
        for (size_t i = 0; i < dcf_parallel; i += 1)
        {
            {
                proc.S[output_base+i][0] = this->fss3prep->rr_share;
                proc.S[output_base+i][1] = this->fss3prep->rr_share_tmp;
            }
        }
        return;
    }
}

template <class T>
void Fss<T>::distributed_comparison_function_gpu(SubProcessor<T> &proc, const Instruction &instruction, int n)
{
}

template <class T>
void Fss<T>::generate()
{
    // 判断buffer里面有没有数
    // 有数值就读出即可
    // 没有数就gen_dcp
}

template <class T>
bigint Fss<T>::evaluate(typename T::open_type x, int n, int result_length, int drop_least_bits)
{
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

    for (int i = 1+drop_least_bits; i < n; i++)
    {
        xi = x.get_bit(n - i - 1);
        prng.SetSeed(seed);
        for (int j = 0; j < 2; j++)
        {
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j], lambda);
            t_hat[j] = s_hat[j].get_ui() & 1;
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->tcw[j][i - 1]);
        }
        if (n <= 128)
        {
            convert[0] = v_hat[0] >> (lambda - result_length);
            convert[1] = v_hat[1] >> (lambda - result_length);
        }

        tmp_v = tmp_v+b * (-1) * (convert[xi]+tmp_t * this->fss3prep->vcw[i - 1])+(1 ^ b) * (convert[xi]+tmp_t * this->fss3prep->vcw[i - 1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
    }

    if (n <= 128)
        convert[0] = s[xi] >> (lambda - result_length);
    tmp_v = tmp_v+b * (-1) * (convert[0]+tmp_t * this->fss3prep->final_cw)+(1 ^ b) * (convert[0]+tmp_t * this->fss3prep->final_cw);
    return tmp_v;
}

template <class T>
bigint Fss<T>::evaluate_conv_relu(typename T::open_type x, int n, int result_length)
{
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
    for (int i = 1; i < n; i++)
    {
        xi = x.get_bit(n - i - 1);
        prng.SetSeed(seed);
        for (int j = 0; j < 2; j++)
        {
            prng.get(v_hat[j], lambda);
            prng.get(s_hat[j], lambda);
            t_hat[j] = s_hat[j].get_ui() & 1;
            s[j] = s_hat[j] ^ (tmp_t * this->fss3prep->full_scw[i - 1]);
            t[j] = t_hat[j] ^ (tmp_t * this->fss3prep->full_tcw[j][i - 1]);
        }

        // std::cout << "v_hat[0], v_hat[1] are " << v_hat[0] << " " << v_hat[1] << std::endl;
        if (n <= 128)
        {
            // std::cout << "length is " << lambda-n << std::endl;
            convert[0] = v_hat[0] >> (lambda - result_length);
            convert[1] = v_hat[1] >> (lambda - result_length);
        }

        // std::cout << "convert[0], convert[1] are " << convert[0] << " " << convert[1] << std::endl;
        tmp_v = tmp_v+b * (-1) * (convert[xi]+tmp_t * this->fss3prep->full_vcw[i - 1])+(1 ^ b) * (convert[xi]+tmp_t * this->fss3prep->full_vcw[i - 1]);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
        // std::cout << tmp_v << std::endl;
    }

    if (n <= 128)
        convert[0] = s[xi] >> (lambda - result_length);
    tmp_v = tmp_v+b * (-1) * (convert[0]+tmp_t * this->fss3prep->full_final_cw)+(1 ^ b) * (convert[0]+tmp_t * this->fss3prep->full_final_cw);
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
    auto add_share = share+tmp[0] - tmp[1];
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
                (r+S[info.source_base+i][0]).pack(cs);
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
                auto c = cs.get<value_type>()+S[info.source_base+i].sum();
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
            S[info.dest_base+i] = c_prime - r_prime;

            if (info.small_gap())
            {
                auto c_dprime = input.finalize(comp_player);
                auto r_msb = input.finalize(gen_player);
                S[info.dest_base+i] += ((r_msb+c_dprime) << (info.k - info.m));
                prepare_mul(r_msb, c_dprime);
            }
        }
    exchange();

    for (auto info : infos)
        for (int i = 0; i < size; i++)
            if (info.small_gap())
                S[info.dest_base+i] -= finalize_mul()
                                        << (info.k - info.m+1);
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
void Fss<T>::fss_cmp(SubProcessor<T> &processor, const Instruction &instruction)
{
    auto &args = instruction.get_start();
    int n = args[3]+args[4] * 2;
    this->init_fss_cmp_prep(processor);
    processor.protocol.distributed_comparison_function(processor, instruction, n);
}

bigint get_mod(bigint x, int k)
{
    bigint res;
    res = x - ((x >> k) << k);
    return res;
}

void bit_transfer(bigint &res, bigint tmp, int cnt)
{   
    res = res + (tmp << cnt);
}

template <class T>
void Conv2dTuple::rfss3_conv2d_trunc_relu_all(SubProcessor<T> &proc, int k, int f, Player &P)
{
    int n = 1+k+2 * f;
    for (int i_batch = 0; i_batch < batch_size; i_batch++)
    {
        size_t base = r1+i_batch * inputs_w * inputs_h * n_channels_in;
        assert(base+inputs_w * inputs_h * n_channels_in <= proc.S.size());
        T *input_base = &proc.S[base];
        for (int out_y = 0; out_y < output_h; out_y++){
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                int in_x_origin = (out_x * stride_w) - padding_w;
                int in_y_origin = (out_y * stride_h) - padding_h;

                for (int filter_y = 0; filter_y < weights_h; filter_y++)
                {
                    int in_y = in_y_origin+filter_y * filter_stride_h;
                    if ((0 <= in_y) and (in_y < inputs_h))
                        for (int filter_x = 0; filter_x < weights_w; filter_x++)
                        {
                            int in_x = in_x_origin+filter_x * filter_stride_w;
                            if ((0 <= in_x) and (in_x < inputs_w))
                            {
                                T *pixel_base = &input_base[(in_y * inputs_w+in_x) * n_channels_in];
                                T *weight_base = &proc.S[r2+(filter_y * weights_w+filter_x) * n_channels_in];
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
    }
    for (int i_batch = 0; i_batch < batch_size; i_batch++)
    {
        size_t base = r0+i_batch * output_h * output_w;
        assert(base+output_h * output_w <= proc.S.size());
        T *output_base = &proc.S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                output_base[out_y * output_w+out_x] =
                    proc.protocol.finalize_dotprod_without_trunc(
                        lengths[i_batch][out_y][out_x]);
            }
    }
}

template <class T>
void Fss<T>::relu_truncs(SubProcessor<T> &proc, const vector<Conv2dTuple> &tuples, const vector<int> &comparison_destination_values, int n, int f)
{
    // this->init_fss_conv_relu_prep(proc, f);
    int batch_size = tuples[0].batch_size, output_h = tuples[0].output_h, output_w = tuples[0].output_w;
    int r0, comp0, mat_size = batch_size * output_h * output_w; 
    for (auto tuple : tuples)
    {
        assert(batch_size == tuple.batch_size);
        assert(output_h == tuple.output_h);
        assert(output_w == tuple.output_w);
    }
    bigint msb_xhat, tmp_select_bit, select_bit[tuples.size() * mat_size], final_result, tmp_reshare;
    typename T::open_type tmp_bit, tmp_sub = (1LL << (n - 1)), masked_x[tuples.size() * mat_size], eval_value[tuples.size() * mat_size];
    octetStream cs, select_cs, reshare_cs;
    if (P.my_num() == EVAL_1 || P.my_num() == EVAL_2)
    {
        for (size_t i = 0; i < tuples.size(); i++)
        {
            r0 = tuples[i].r0;
            for (int j = 0; j < mat_size; j++)
            {
                proc.S[j+r0][0] = proc.S[j+r0][0]+this->fss3prep->r_mask_share;
                proc.S[j+r0][0].pack(cs);
            }
        }
        P.send_to((P.my_num() ^ 1), cs);
        P.receive_player((P.my_num() ^ 1), cs);
        for (size_t i = 0; i < tuples.size(); i++)
        {
            r0 = tuples[i].r0;
            for (int j = 0; j < mat_size; j++)
            {
                masked_x[i*mat_size+j].unpack(cs);
                proc.S[j+r0][0] = proc.S[j+r0][0]+masked_x[i*mat_size+j];
                masked_x[i*mat_size+j] = proc.S[j+r0][0];
                msb_xhat = masked_x[i*mat_size+j].get_bit(n - 1);
                if (msb_xhat)
                {
                    eval_value[i*mat_size+j] = masked_x[i*mat_size+j] - ((masked_x[i*mat_size+j] >> (n - 1)) << (n - 1));
                }
                else
                {
                    eval_value[i*mat_size+j] = masked_x[i*mat_size+j];
                }
                eval_value[i*mat_size+j] = tmp_sub - eval_value[i*mat_size+j] - 1;
                select_bit[i*mat_size+j] = bigint(typename T::open_type(this->evaluate_conv_relu(eval_value[i*mat_size+j], n, 1)));
                select_bit[i*mat_size+j] = (select_bit[i*mat_size+j] ^ this->fss3prep->r_drelu_share) ^ (P.my_num() * msb_xhat);
                select_bit[i*mat_size+j] = select_bit[i*mat_size+j].get_ui() % 2;
                select_bit[i*mat_size+j] = select_bit[i*mat_size+j] ^ this->fss3prep->r_select_share;
                bit_transfer(tmp_select_bit, select_bit[i*mat_size+j], i * mat_size+j);
            }
        }
        tmp_select_bit.pack(select_cs);
        P.send_to((P.my_num() ^ 1), select_cs);
        P.receive_player((P.my_num() ^ 1), select_cs);
        tmp_select_bit.unpack(select_cs);
        for (size_t i = 0; i < tuples.size(); i++){
            r0 = tuples[i].r0;
            for (int j = 0; j < mat_size; j++)
            {
                select_bit[i*mat_size+j] = select_bit[i*mat_size+j] ^ ((tmp_select_bit>>(i*mat_size+j))%2);
                tmp_bit = (masked_x[i*mat_size+j] >> (n - 1));
                if (select_bit[i*mat_size+j].get_ui())
                {
                    if (tmp_bit.get_bit(0))
                    {
                        final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i*mat_size+j], n) >> f), n - f - 1) - this->fss3prep->o_select_share+this->fss3prep->v_select_share+((this->fss3prep->reverse_1_u_select_share) << (n - f - 1));
                    }
                    else
                    {
                        final_result = (P.my_num() - this->fss3prep->u_select_share) * get_mod((get_mod(masked_x[i*mat_size+j], n) >> f), n - f - 1) - this->fss3prep->o_select_share+this->fss3prep->v_select_share+((this->fss3prep->p_select_share - this->fss3prep->w_select_share) << (n - f - 1));
                    }
                }
                else
                {
                    if (tmp_bit.get_bit(0))
                    {
                        final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i*mat_size+j], n) >> f), n - f - 1) - this->fss3prep->v_select_share+(this->fss3prep->reverse_u_select_share << (n - f - 1));
                    }
                    else
                    {
                        final_result = this->fss3prep->u_select_share * get_mod((get_mod(masked_x[i*mat_size+j], n) >> f), n - f - 1) - this->fss3prep->v_select_share+(this->fss3prep->w_select_share << (n - f - 1));
                    }
                }
                proc.S[r0+j][(P.my_num()) ^ 1] = this->fss3prep->reshare_value;
                tmp_reshare = final_result - this->fss3prep->reshare_value;
                proc.S[r0+j][(P.my_num())] = tmp_reshare;
                tmp_reshare.pack(reshare_cs);
            }
        }  
        P.send_to((P.my_num()^1), reshare_cs);
        P.receive_player((P.my_num()^1), reshare_cs);
        for (size_t i = 0; i < tuples.size(); i++){
            r0 = tuples[i].r0;
            comp0 = comparison_destination_values[i];
            for (int j = 0; j < mat_size; j++)
            {
                this->fss3prep->reshare_value_tmp.unpack(reshare_cs);
                proc.S[r0+j][(P.my_num())] = proc.S[r0+j][(P.my_num())]+this->fss3prep->reshare_value_tmp;
                if(P.my_num() == GEN)
                {
                    proc.S[comp0+j][(P.my_num())] = 0;
                }
                else{
                    proc.S[comp0+j][(P.my_num())] = select_bit[i*mat_size+j];
                    std::cout << "select_bit is " << select_bit[i*mat_size+j]<< std::endl;
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < tuples.size(); i++){
            r0 = tuples[i].r0;
            for (int j = 0; j < mat_size; j++)
            {
                proc.S[r0+j][0] = this->fss3prep->reshare_value;
                proc.S[r0+j][1] = this->fss3prep->reshare_value_tmp;
            }
        }
    }
}

// instructions for rfss3
template <class T>
void Fss<T>::rfss3s_conv2d_relu_truncs(SubProcessor<T> &proc, const Instruction &instruction)
{
    std::cout << "calling rfss3s_conv2d_relu_truncs" << std::endl;
    // octetStream cs;
    auto &args = instruction.get_start();
    proc.protocol.init_dotprod_without_trunc();
    vector<Conv2dTuple> tuples;
    vector<int>comparison_destination_values;
    int n = 1+args[16]+2 * args[17], k = args[16], f = args[17];
    if (this->fss3prep == nullptr)
    {
        DataPositions usage;
        this->fss3prep = new typename T::LivePrep(&proc, usage);        
        // if (P.my_num() == GEN)
        // {
        //     this->fss3prep->gen_fake_conv_relu(1, n, f);
        //     signal = 1;
        //     signal.pack(cs);
        //     P.send_to(EVAL_1, cs);
        //     P.send_to(EVAL_2, cs);
        // }
        // else
        // {
        //     P.receive_player(GEN, cs);
        //     signal.unpack(cs);
        // }
        // this->fss3prep->init_offline_values(&proc, 1);
    }
    
    for (size_t i = 0; i < args.size(); i += 19)
    {
        tuples.push_back(Conv2dTuple(args, i));
        comparison_destination_values.push_back(args[i+18]);
        assert(n == 1+args[i+16]+2 * args[i+17]);
    }
    for (size_t i = 0; i < tuples.size(); i++)
    {
        tuples[i].rfss3_conv2d_trunc_relu_all(proc, k, f, P);
    }
    relu_truncs(proc, tuples, comparison_destination_values, n, f);
    return;
}

#endif // PROTOCOLS_FSS_HPP