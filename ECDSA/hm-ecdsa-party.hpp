/*
 * mal-rep-ecdsa-party.cpp
 *
 */

#include "Networking/Server.h"
#include "Networking/CryptoPlayer.h"
#include "Protocols/Replicated.h"
#include "Protocols/MaliciousRep3Share.h"
#include "Protocols/ReplicatedInput.h"
#include "Math/gfp.h"
#include "ECDSA/P256Element.h"
#include "Tools/Bundle.h"
#include "GC/TinyMC.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/CcdSecret.h"
#include "GC/VectorInput.h"

#include "ECDSA/preprocessing.hpp"
#include "ECDSA/sign.hpp"
#include "Protocols/MaliciousRepMC.hpp"
#include "Protocols/Beaver.hpp"
#include "Protocols/fake-stuff.hpp"
#include "Protocols/MaliciousRepPrep.hpp"
#include "Processor/Input.hpp"
#include "Processor/Processor.hpp"
#include "Processor/Data_Files.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/RepPrep.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Secret.hpp"
#include "Machines/ShamirMachine.hpp"
#include "Machines/MalRep.hpp"
#include "Machines/Rep.hpp"

#include <assert.h>

template<template<class U> class T>
void run(int argc, const char** argv)
{
    bigint::init_thread();
    ez::ezOptionParser opt;
    EcdsaOptions opts(opt, argc, argv);
    Names N(opt, argc, argv, 3);
    int n_tuples = 1000;
    if (not opt.lastArgs.empty())
        n_tuples = atoi(opt.lastArgs[0]->c_str());
    CryptoPlayer P(N, "ecdsa");
    P256Element::init();
    typedef T<P256Element::Scalar> pShare;
    OnlineOptions::singleton.batch_size = 1;
    // synchronize
    Bundle<octetStream> bundle(P);
    P.unchecked_broadcast(bundle);
    Timer timer;
    timer.start();
    auto stats = P.total_comm();
    pShare sk = typename T<P256Element::Scalar>::Honest::Protocol(P).get_random();
    cout << "Secret key generation took " << timer.elapsed() * 1e3 << " ms" << endl;
    (P.total_comm() - stats).print(true);

    OnlineOptions::singleton.batch_size = (1 + pShare::Protocol::uses_triples) * n_tuples;
    DataPositions usage;
    typename pShare::TriplePrep prep(0, usage);
    typename pShare::MAC_Check MCp;
    ArithmeticProcessor _({}, 0);
    SubProcessor<pShare> proc(_, MCp, prep, P);

    bool prep_mul = not opt.isSet("-D");
    vector<EcTuple<T>> tuples;
    preprocessing(tuples, n_tuples, sk, proc, opts);
//    check(tuples, sk, {}, P);
    sign_benchmark(tuples, sk, MCp, P, opts, prep_mul ? 0 : &proc);
    P256Element::finish();
}
