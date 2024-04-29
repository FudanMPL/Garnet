/*
 * aby2ed.cpp
 *
 */

#ifndef PROTOCOLS_ABY2ED_HPP_
#define PROTOCOLS_ABY2ED_HPP_

#include "aby2ed.h"
#include "Processor/Processor.h"
#include "Processor/TruncPrTuple.h"
#include "Tools/benchmarking.h"
#include "Tools/Bundle.h"

#include "ReplicatedInput.h"
#include "Rep3Share2k.h"
#include "aby2Input.h"
#include "ReplicatedPO.hpp"
#include "Math/Z2k.hpp"

template<class T>
aby2ProtocolBase<T>::aby2ProtocolBase() :
        trunc_pr_counter(0), rounds(0), trunc_rounds(0), dot_counter(0),
        bit_counter(0), counter(0)
{
}

template<class T>
aby2ed<T>::aby2ed(Player& P) : aby2edBase(P)
{
    assert(T::vector_length == 2);
}

template<class T>
aby2ed<T>::aby2ed(const aby2edBase& other) :
        aby2edBase(other)
{
}

inline aby2edBase::aby2edBase(Player& P) : P(P)
{
    // assert(P.num_players() == 3);
	if (not P.is_encrypted())
		insecure("unencrypted communication", false);
    octetStream os;
    if(P.my_num()==0)
    {
        shared_prngs[0].ReSeed();
        shared_prngs[1].ReSeed();
	    os.append(shared_prngs[0].get_seed(), SEED_SIZE);
	    P.send_relative(1, os); //发送给1方
    }
    else
    {
        shared_prngs[1].ReSeed();
        P.receive_relative(-1, os);
        shared_prngs[0].SetSeed(os.get_data());
    }
	
}

inline aby2edBase::aby2edBase(Player& P, array<PRNG, 2>& prngs) :
        P(P)
{
    for (int i = 0; i < 2; i++)
        shared_prngs[i].SetSeed(prngs[i]);
}

// inline aby2edBase aby2edBase::branch() const
// {
//     return {P, shared_prngs};
// }

template<class T>


aby2ProtocolBase<T>::~aby2ProtocolBase()
{
#ifdef VERBOSE_COUNT
    if (counter or rounds)
        cerr << "Number of " << T::type_string() << " multiplications: "
                << counter << " (" << bit_counter << " bits) in " << rounds
                << " rounds" << endl;
    if (counter or rounds)
        cerr << "Number of " << T::type_string() << " dot products: " << dot_counter << endl;
    if (trunc_pr_counter or trunc_rounds)
        cerr << "Number of probabilistic truncations: " << trunc_pr_counter << " in " << trunc_rounds << " rounds" << endl;
#endif
}

template<class T>
void aby2ProtocolBase<T>::muls(const vector<int>& reg,
        SubProcessor<T>& proc, typename T::MAC_Check& MC, int size)
{
    (void)MC;
    proc.muls(reg, size);
}

template<class T>
void aby2ProtocolBase<T>::mulrs(const vector<int>& reg,
        SubProcessor<T>& proc)
{
    proc.mulrs(reg);
}

template<class T>
void aby2ProtocolBase<T>::multiply(vector<T>& products,
        vector<pair<T, T> >& multiplicands, int begin, int end,
        SubProcessor<T>& proc)
{
#ifdef VERBOSE_CENTRAL
    fprintf(stderr, "multiply from %d to %d in %d\n", begin, end,
            BaseMachine::thread_num);
#endif
    init(proc.DataF, proc.MC);
    init_mul();
    for (int i = begin; i < end; i++)
        prepare_mul(multiplicands[i].first, multiplicands[i].second);
    exchange();
    for (int i = begin; i < end; i++)
        products[i] = finalize_mul();
}

template<class T>
T aby2ProtocolBase<T>::mul(const T& x, const T& y)
{
    init_mul();
    prepare_mul(x, y);
    exchange();
    return finalize_mul();
}

template<class T>
void aby2ProtocolBase<T>::prepare_mult(const T& x, const T& y, int n,
		bool)
{
    prepare_mul(x, y, n);
}

template<class T>
void aby2ProtocolBase<T>::finalize_mult(T& res, int n)
{
    res = finalize_mul(n);
}

template<class T>
T aby2ProtocolBase<T>::finalize_dotprod(int length)
{
    counter += length;
    dot_counter++;
    T res;
    for (int i = 0; i < length; i++)
        res += finalize_mul();
    return res;
}

template<class T>
T aby2ProtocolBase<T>::get_random()
{
    if (random.empty())
    {
        buffer_random();
        assert(not random.empty());
    }

    auto res = random.back();
    random.pop_back();
    return res;
}

template<class T>
vector<int> aby2ProtocolBase<T>::get_relevant_players()
{
    vector<int> res;
    int n = dynamic_cast<typename T::Protocol&>(*this).P.num_players();
    for (int i = 0; i < T::threshold(n) + 1; i++)
        res.push_back(i);
    return res;
}

template<class T>
void aby2ed<T>::init_mul()
{
    cout<<"进入init_mul"<<endl;
    for (auto& o : os)
        o.reset_write_head();
    os_vec.resize(2);
    for (auto& o : os_vec)
        o.reset_write_head();
    add_shares.clear();
    // assert(this->prep);
    // assert(this->MC);
    shares.clear();
    opened.clear();
    triples.clear();
    lengths.clear();
    delta_y.clear();
      cout<<__LINE__<<endl;
    delta_y.push_back({});
    cout<<__LINE__<<endl;
    delta_y.back().randomize(shared_prngs[1],0);
}

template<class T>
void aby2ed<T>::prepare_mul(const T& x,
        const T& y, int n)
{
    // cout<<__LINE__<<endl;
    // typename T::value_type add_share = x.local_mul(y);
    // prepare_reshare(add_share, n);
    typename T::value_type a=0;
    if(P.my_num())
        a=x[0].lazy_mul(y[0]);
    T triple_delta_c=0;//这里应该是从离线阶段获取到三元组的，但是这里进行了设置为0的操作，所以得到的结果中Delta值一定不争取
    auto b=a-x[0]*y[1]+x[1]*y[0]+delta_y.back();
//   cout<<__LINE__<<endl;
    // triples.push_back({{}});
    // auto& triple = triples.back();
    // triple = prep->get_triple(n); //这个函数需要自己在进行按需要重新实现
    // triple={1,2,3};
    shares.push_back(b);
    //   cout<<__LINE__<<endl;
    b.pack(os_vec[P.my_num()], n);
    //   cout<<__LINE__<<endl;
    lengths.push_back(n);
    // cout<<'\n'<<P.my_num()<<"方在prepare_mul函数中，最后得到的[Delta_y]的share为："<<shares.back()<<endl;
}


template<class T>
void aby2ed<T>::prepare_reshare(const typename T::clear& share,
        int n)//目前不太需要了
{
    typename T::value_type tmp[2];
    for (int i = 0; i < 2; i++)
        tmp[i].randomize(shared_prngs[i], n);
    auto add_share = share + tmp[0] - tmp[1];
    add_share.pack(os[0], n);
    add_shares.push_back(add_share);
    
}

template<class T>
void aby2ed<T>::exchange()
{
    cout<<"In exchange : size="<<shares.size()<<endl;
    int mynum=P.my_num();
    P.send_to(1-mynum,os_vec[mynum]);

    auto& dest =  os_vec[1-mynum];//
    P.receive_player(1-mynum , dest);
}

template<class T>
void aby2ed<T>::start_exchange()
{
    P.send_relative(1, os[0]);
    this->rounds++;
}

template<class T>
void aby2ed<T>::stop_exchange()
{
    P.receive_relative(-1, os[1]);
}

template<class T>
inline T aby2ed<T>::finalize_mul(int n)
{
     (void) n;
    typename T::value_type t;//这里的t类型为Z2<K>
    t.unpack(os_vec[1-P.my_num()], n);
    opened.push_back(t+shares.back());
    cout<<"在exchange函数中，最后得到的O进行open操作Delta_y后的结果为："<<opened[0]<<endl;//

    typename T::open_type masked[2];
    T result;
    result[0]=opened[0];
    result[1]=delta_y.back();
    cout<<"在finalize_mul函数中，最后得到的y的share为："<<result[0]<<'\t'<<result[1]<<endl;
    return result;
}

template<class T>
inline void aby2ed<T>::init_dotprod()
{
    init_mul();
    dotprod_share.assign_zero();
}

template<class T>
inline void aby2ed<T>::prepare_dotprod(const T& x, const T& y)
{
    prepare_mul(x,y);
    dotprod_share = dotprod_share.lazy_add(shares.back());
}

template<class T>
inline void aby2ed<T>::next_dotprod()
{
    dotprod_share.normalize();
    // prepare_reshare(dotprod_share);
    dotprod_share.pack(os_vec[P.my_num()]);
    dotprod_share.assign_zero();
}

template<class T>
inline T aby2ed<T>::finalize_dotprod(int length)
{
    (void) length;
    this->dot_counter++;
    return finalize_mul();
}

template<class T>
T aby2ed<T>::get_random()
{
    T res;
    for (int i = 0; i < 2; i++)
        res[i].randomize(shared_prngs[i]);
    return res;
}

template<class T>
void aby2ProtocolBase<T>::randoms_inst(vector<T>& S,
		const Instruction& instruction)
{
    for (int j = 0; j < instruction.get_size(); j++)
    {
        auto& res = S[instruction.get_r(0) + j];
        randoms(res, instruction.get_n());
    }
}

template<class T>
void aby2ed<T>::randoms(T& res, int n_bits)
{
    for (int i = 0; i < 2; i++)
        res[i].randomize_part(shared_prngs[i], n_bits);
}

template<class T>
template<class U>
void aby2ed<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
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
    auto& S = proc.get_S();

    octetStream cs;
    aby2Input<T> input(0, *this);

    // use https://eprint.iacr.org/2019/131
    bool have_small_gap = false;
    // use https://eprint.iacr.org/2018/403
    bool have_big_gap = false;

    for (auto info : infos)
        if (info.small_gap())
            have_small_gap = true;
        else
            have_big_gap = true;

    if (generate)
    {
        SeededPRNG G;
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto& x = S[info.source_base + i];
                if (info.small_gap())
                {
                    auto r = G.get<value_type>();
                    input.add_mine(info.upper(r));
                    input.add_mine(info.msb(r));
                    (r + x[0]).pack(cs);
                }
                else
                {
                    auto& y = S[info.dest_base + i];
                    auto r = this->shared_prngs[0].template get<value_type>();
                    y[1] = -value_type(-value_type(x.sum()) >> info.m) - r;
                    y[1].pack(cs);
                    y[0] = r;
                }
            }

        P.send_to(comp_player, cs);
    }
    else if (have_small_gap)
        input.add_other(gen_player);

    if (compute)
    {
        P.receive_player(gen_player, cs);
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto& x = S[info.source_base + i];
                if (info.small_gap())
                {
                    auto c = cs.get<value_type>() + x.sum();
                    input.add_mine(info.upper(c));
                    input.add_mine(info.msb(c));
                }
                else
                {
                    auto& y = S[info.dest_base + i];
                    y[0] = cs.get<value_type>();
                    y[1] = x[1] >> info.m;
                }
            }
    }

    if (have_big_gap and not (compute or generate))
    {
        for (auto info : infos)
            if (info.big_gap())
                for (int i = 0; i < size; i++)
                {
                    auto& x = S[info.source_base + i];
                    auto& y = S[info.dest_base + i];
                    y[0] = x[0] >> info.m;
                    y[1] = this->shared_prngs[1].template get<value_type>();
                }
    }

    if (have_small_gap)
    {
        input.add_other(comp_player);
        input.exchange();
        init_mul();

        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                if (info.small_gap())
                {
                    this->trunc_pr_counter++;
                    auto c_prime = input.finalize(comp_player);
                    auto r_prime = input.finalize(gen_player);
                    S[info.dest_base + i] = c_prime - r_prime;

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
}

template<class T>
template<class U>
void aby2ed<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
        true_type)
{
    (void) regs, (void) size, (void) proc;
    throw runtime_error("trunc_pr not implemented");
}

template<class T>
template<class U>
void aby2ed<T>::trunc_pr(const vector<int>& regs, int size,
        U& proc)
{
    this->trunc_rounds++;
    trunc_pr(regs, size, proc, T::clear::characteristic_two);
}

#endif
