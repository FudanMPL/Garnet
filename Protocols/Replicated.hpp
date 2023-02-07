/*
 * Replicated.cpp
 *
 */

#ifndef PROTOCOLS_REPLICATED_HPP_
#define PROTOCOLS_REPLICATED_HPP_

#include "Replicated.h"
#include "Processor/Processor.h"
#include "Processor/TruncPrTuple.h"
#include "Tools/benchmarking.h"
#include "Tools/Bundle.h"

#include "ReplicatedInput.h"
#include "Rep3Share2k.h"

#include "ReplicatedPO.hpp"
#include "Math/Z2k.hpp"

template<class T>
ProtocolBase<T>::ProtocolBase() :
        trunc_pr_counter(0), rounds(0), trunc_rounds(0), dot_counter(0),
        bit_counter(0), counter(0)
{
}

template<class T>
Replicated<T>::Replicated(Player& P) : ReplicatedBase(P)
{
    assert(T::vector_length == 2);
}

template<class T>
Replicated<T>::Replicated(const ReplicatedBase& other) :
        ReplicatedBase(other)
{
}

inline ReplicatedBase::ReplicatedBase(Player& P) : P(P)
{
    assert(P.num_players() == 3);
	if (not P.is_encrypted())
		insecure("unencrypted communication", false);

	shared_prngs[0].ReSeed();
	octetStream os;
	os.append(shared_prngs[0].get_seed(), SEED_SIZE);
	P.send_relative(1, os);
	P.receive_relative(-1, os);
	shared_prngs[1].SetSeed(os.get_data());
}

inline ReplicatedBase::ReplicatedBase(Player& P, array<PRNG, 2>& prngs) :
        P(P)
{
    for (int i = 0; i < 2; i++)
        shared_prngs[i].SetSeed(prngs[i]);
}

inline ReplicatedBase ReplicatedBase::branch()
{
    return {P, shared_prngs};
}

template<class T>
ProtocolBase<T>::~ProtocolBase()
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
void ProtocolBase<T>::muls(const vector<int>& reg,
        SubProcessor<T>& proc, typename T::MAC_Check& MC, int size)
{
    (void)MC;
    proc.muls(reg, size);
}

template<class T>
void ProtocolBase<T>::mulrs(const vector<int>& reg,
        SubProcessor<T>& proc)
{
    proc.mulrs(reg);
}

template<class T>
void ProtocolBase<T>::multiply(vector<T>& products,
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
T ProtocolBase<T>::mul(const T& x, const T& y)
{
    init_mul();
    prepare_mul(x, y);
    exchange();
    return finalize_mul();
}

template<class T>
void ProtocolBase<T>::prepare_mult(const T& x, const T& y, int n,
		bool)
{
    prepare_mul(x, y, n);
}

template<class T>
void ProtocolBase<T>::finalize_mult(T& res, int n)
{
    res = finalize_mul(n);
}

template<class T>
T ProtocolBase<T>::finalize_dotprod(int length)
{
    counter += length;
    dot_counter++;
    T res;
    for (int i = 0; i < length; i++)
        res += finalize_mul();
    return res;
}

template<class T>
T ProtocolBase<T>::get_random()
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
vector<int> ProtocolBase<T>::get_relevant_players()
{
    vector<int> res;
    int n = dynamic_cast<typename T::Protocol&>(*this).P.num_players();
    for (int i = 0; i < T::threshold(n) + 1; i++)
        res.push_back(i);
    return res;
}

template<class T>
void Replicated<T>::init_mul()
{
    for (auto& o : os)
        o.reset_write_head();
    add_shares.clear();
}

template<class T>
void Replicated<T>::prepare_mul(const T& x,
        const T& y, int n)
{
    typename T::value_type add_share = x.local_mul(y);
    prepare_reshare(add_share, n);
}

template<class T>
void Replicated<T>::prepare_reshare(const typename T::clear& share,
        int n)
{
    typename T::value_type tmp[2];
    for (int i = 0; i < 2; i++)
        tmp[i].randomize(shared_prngs[i], n);
    auto add_share = share + tmp[0] - tmp[1];
    add_share.pack(os[0], n);
    add_shares.push_back(add_share);
}

template<class T>
void Replicated<T>::exchange()
{
    if (os[0].get_length() > 0)
        P.pass_around(os[0], os[1], 1);
    this->rounds++;
}

template<class T>
void Replicated<T>::start_exchange()
{
    P.send_relative(1, os[0]);
    this->rounds++;
}

template<class T>
void Replicated<T>::stop_exchange()
{
    P.receive_relative(-1, os[1]);
}

template<class T>
inline T Replicated<T>::finalize_mul(int n)
{
    this->counter++;
    this->bit_counter += n;
    T result;
    result[0] = add_shares.next();
    result[1].unpack(os[1], n);
    return result;
}

template<class T>
inline void Replicated<T>::init_dotprod()
{
    init_mul();
    dotprod_share.assign_zero();
}

template<class T>
inline void Replicated<T>::prepare_dotprod(const T& x, const T& y)
{
    dotprod_share = dotprod_share.lazy_add(x.local_mul(y));
}

template<class T>
inline void Replicated<T>::next_dotprod()
{
    dotprod_share.normalize();
    prepare_reshare(dotprod_share);
    dotprod_share.assign_zero();
}

template<class T>
inline T Replicated<T>::finalize_dotprod(int length)
{
    (void) length;
    this->dot_counter++;
    return finalize_mul();
}

template<class T>
T Replicated<T>::get_random()
{
    T res;
    for (int i = 0; i < 2; i++)
        res[i].randomize(shared_prngs[i]);
    return res;
}

template<class T>
void ProtocolBase<T>::randoms_inst(vector<T>& S,
		const Instruction& instruction)
{
    for (int j = 0; j < instruction.get_size(); j++)
    {
        auto& res = S[instruction.get_r(0) + j];
        randoms(res, instruction.get_n());
    }
}

template<class T>
void Replicated<T>::randoms(T& res, int n_bits)
{
    for (int i = 0; i < 2; i++)
        res[i].randomize_part(shared_prngs[i], n_bits);
}

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
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

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
        true_type)
{
    (void) regs, (void) size, (void) proc;
    throw runtime_error("trunc_pr not implemented");
}

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size,
        U& proc)
{
    this->trunc_rounds++;
    trunc_pr(regs, size, proc, T::clear::characteristic_two);
}

#endif
