/*
 * preprocessing.hpp
 *
 */

#ifndef ECDSA_PREPROCESSING_HPP_
#define ECDSA_PREPROCESSING_HPP_

#include "P256Element.h"
#include "EcdsaOptions.h"
#include "Processor/Data_Files.h"
#include "Protocols/ReplicatedPrep.h"
#include "Protocols/MaliciousShamirShare.h"
#include "Protocols/Rep3Share.h"
#include "GC/TinierSecret.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/TinyMC.h"

#include "GC/TinierSharePrep.hpp"
#include "GC/CcdSecret.h"

template<template<class U> class T>
class EcTuple
{
public:
    T<P256Element::Scalar> a;
    T<P256Element::Scalar> b;
    P256Element::Scalar c;
    T<P256Element> secret_R;
    P256Element R;
};

template<template<class U> class T>
void preprocessing(vector<EcTuple<T>>& tuples, int buffer_size,
        T<P256Element::Scalar>& sk,
        SubProcessor<T<P256Element::Scalar>>& proc,
        EcdsaOptions opts)
{
    bool prep_mul = opts.prep_mul;
    Timer timer;
    timer.start();
    Player& P = proc.P;
    auto& prep = proc.DataF;
    size_t start = P.total_comm().sent;
    auto stats = P.total_comm();
    auto& extra_player = P;

    auto& protocol = proc.protocol;
    auto& MCp = proc.MC;
    typedef T<typename P256Element::Scalar> pShare;
    typedef T<P256Element> cShare;
    vector<pShare> inv_ks;
    vector<cShare> secret_Rs;
    prep.buffer_triples();
    vector<pShare> bs, cs;
    for (int i = 0; i < buffer_size; i++)
    {
        pShare a, b, c;
        prep.get_three(DATA_TRIPLE, a, b, c);
        inv_ks.push_back(a);
        bs.push_back(b);
        cs.push_back(c);
    }
    vector<P256Element::Scalar> cs_opened;
    MCp.POpen_Begin(cs_opened, cs, extra_player);
    if (opts.fewer_rounds)
        secret_Rs.insert(secret_Rs.begin(), bs.begin(), bs.end());
    else
    {
        MCp.POpen_End(cs_opened, cs, extra_player);
        for (int i = 0; i < buffer_size; i++)
            secret_Rs.push_back(bs[i] / cs_opened[i]);
    }
    vector<P256Element> opened_Rs(buffer_size);
    typename cShare::Direct_MC MCc(MCp.get_alphai());
    if (not opts.R_after_msg)
        MCc.POpen_Begin(opened_Rs, secret_Rs, extra_player);
    if (prep_mul)
    {
        protocol.init_mul();
        for (int i = 0; i < buffer_size; i++)
            protocol.prepare_mul(inv_ks[i], sk);
        protocol.start_exchange();
    }
    if (opts.fewer_rounds)
        MCp.POpen_End(cs_opened, cs, extra_player);
    if (not opts.R_after_msg)
    {
        MCc.POpen_End(opened_Rs, secret_Rs, extra_player);
        if (opts.fewer_rounds)
            for (int i = 0; i < buffer_size; i++)
                opened_Rs[i] /= cs_opened[i];
    }
    if (prep_mul)
        protocol.stop_exchange();
    if (opts.check_open)
        MCc.Check(extra_player);
    if (opts.check_open or opts.check_beaver_open)
        MCp.Check(extra_player);
    for (int i = 0; i < buffer_size; i++)
    {
        tuples.push_back(
                { inv_ks[i], prep_mul ? protocol.finalize_mul() : pShare(),
                        cs_opened[i], secret_Rs[i], opened_Rs[i] });
    }
    timer.stop();
    cout << "Generated " << buffer_size << " tuples in " << timer.elapsed()
            << " seconds, throughput " << buffer_size / timer.elapsed() << ", "
            << 1e-3 * (P.total_comm().sent - start) / buffer_size
            << " kbytes per tuple" << endl;
    (P.total_comm() - stats).print(true);
}

template<template<class U> class T>
void check(vector<EcTuple<T>>& tuples, T<P256Element::Scalar> sk,
    P256Element::Scalar alphai, Player& P)
{
    typename T<P256Element::Scalar>::MAC_Check MC(alphai);
    auto open_sk = MC.open(sk, P);
    for (auto& tuple : tuples)
    {
        auto inv_k = MC.open(tuple.a, P);
        auto k = inv_k.invert();
        assert(open_sk * inv_k == MC.open(tuple.b, P));
        assert(tuple.R == k);
    }
    MC.Check(P);
}

#endif /* ECDSA_PREPROCESSING_HPP_ */
