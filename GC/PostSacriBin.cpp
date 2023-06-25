/*
 * Abfllnoww.cpp
 *
 */

#include "PostSacriBin.h"

#include "Processor/Processor.h"

#include "Protocols/Replicated.hpp"
#include "Protocols/MaliciousRepMC.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "ShareSecret.hpp"

namespace GC
{

PostSacriBin::PostSacriBin(Player& P) :
        ReplicatedBase(P), honest(P)
{
}

PostSacriBin::~PostSacriBin()
{
    if (not inputs.empty())
    {
        cerr << "unchecked ANDs" << endl;
        terminate();
    }
}

void PostSacriBin::init_mul()
{
    if ((int) inputs.size() >= OnlineOptions::singleton.batch_size)
        check();
    honest.init_mul();
}

void PostSacriBin::prepare_mul(const T& x, const T& y, int n)
{
    if (n == -1)
        n = T::default_length;
    honest.prepare_mul(x, y, n);
    inputs.push_back({{x.mask(n), y.mask(n)}});
}

void PostSacriBin::exchange()
{
    honest.exchange();
}

PostSacriBin::T PostSacriBin::finalize_mul(int n)
{
    if (n == -1)
        n = T::default_length;
    auto res = honest.finalize_mul(n);
    outputs.push_back({res, n});
    return res;
}

void PostSacriBin::check()
{
    vector<array<T, 3>> to_check;
    assert(inputs.size() == outputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        to_check.push_back({{inputs[i][0], inputs[i][1], outputs[i].first}});
    GlobalPRNG G(P);
    for (size_t i = 0; i < inputs.size(); i++)
        to_check.push_back(get_d1_triple(G, outputs[i].second));
    HashMaliciousRepMC<T> MC;
    vector<array<T, 3>> _(N);
    TripleShuffleSacrifice<T>(2, 6).triple_sacrifice(_, to_check, P, MC, 0, inputs.size());
    MC.Check(P);
    inputs.clear();
    outputs.clear();
}

array<PostSacriBin::T, 3> PostSacriBin::get_d1_triple(GlobalPRNG& G, int n_bits)
{
    while (d1.size() < N)
        d1.push_back(get_d2_triple(T::N_BITS));
    int i = G.get_uint(N);
    auto tmp = d1.at(i).mask(n_bits);
    d1[i] <<= n_bits;
    d1[i] ^= get_d2_triple(n_bits);
    array<T, 3> res({{tmp[0], tmp[1], tmp[2]}});
    return res;
}

array<PostSacriBin::T, 3> PostSacriBin::get_d2_triple(int n_bits)
{
    return get_triple_no_count(n_bits);
}

void PostSacriBin::get(Dtype type, T* res)
{
    assert(type == DATA_TRIPLE);

    if (d2.empty())
    {
        TripleShuffleSacrifice<T> sacrifice(2, 6);
        vector<array<T, 3>> check_triples;

        // optimistic triple generation
        Replicated<T> protocol(P);
        generate_triples(check_triples, 2 * N + 6, &protocol, T::N_BITS);
        HashMaliciousRepMC<T> MC;
        sacrifice.triple_sacrifice(d2, check_triples, P, MC, 0);
        MC.Check(P);
        assert(d2.size() == N);
    }

    for (int i = 0; i < 3; i++)
        res[i] = d2.back()[i];
    d2.pop_back();
}

} /* namespace GC */
