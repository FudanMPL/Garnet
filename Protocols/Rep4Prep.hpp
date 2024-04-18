/*
 * Rep4Prep.hpp
 *
 */

#ifndef PROTOCOLS_REP4PREP_HPP_
#define PROTOCOLS_REP4PREP_HPP_

#include "Rep4Prep.h"

template<class T>
Rep4RingPrep<T>::Rep4RingPrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage), BitPrep<T>(proc, usage),
        RingPrep<T>(proc, usage),
        MaliciousDabitOnlyPrep<T>(proc, usage),
        MaliciousRingPrep<T>(proc, usage)
{
}

template<class T>
Rep4RingOnlyPrep<T>::Rep4RingOnlyPrep(SubProcessor<T>* proc,
        DataPositions& usage) :
        BufferPrep<T>(usage), BitPrep<T>(proc, usage),
        RingPrep<T>(proc, usage),
        MaliciousDabitOnlyPrep<T>(proc, usage),
        Rep4RingPrep<T>(proc, usage),
        RepRingOnlyEdabitPrep<T>(proc, usage)
{
}

template<class T>
void Rep4RingPrep<T>::buffer_inputs(int player)
{
    auto prot = this->protocol;
    assert(prot != 0);
    for (int i = 0; i < OnlineOptions::singleton.batch_size; i++)
    {
        T res;
        for (int j = 0; j < 3; j++)
            if (prot->P.get_offset(player - j) != 1)
                res[j].randomize(prot->rep_prngs[j]);
        this->inputs[player].push_back({res, res.sum()});
    }
}

template<class T>
void Rep4RingPrep<T>::buffer_triples()
{
    generate_triples(this->triples, OnlineOptions::singleton.batch_size,
            this->protocol);
}

template<class T>
void Rep4RingPrep<T>::buffer_squares()
{
    generate_squares(this->squares, OnlineOptions::singleton.batch_size,
            this->protocol);
}

template<class T>
void Rep4RingPrep<T>::buffer_bits()
{
    assert(this->proc != 0);
    SeededPRNG G;
    octetStream os;
    Player& P = this->proc->P;
    if (P.my_num() % 2 == 0)
    {
        os.append(G.get_seed(), SEED_SIZE);
        P.send_relative(1, os);
    }
    else
    {
        P.receive_relative(-1, os);
        G.SetSeed(os.consume(SEED_SIZE));
    }

    auto& protocol = this->proc->protocol;

    vector<typename T::open_type> bits;
    int batch_size = OnlineOptions::singleton.batch_size;
    bits.reserve(batch_size);
    for (int i = 0; i < batch_size; i++)
        bits.push_back(G.get_bit());

    protocol.init_mul();
    protocol.reset_joint_input(batch_size);
    vector<typename Rep4<T>::ResTuple> a(batch_size), b(batch_size);
    protocol.prepare_joint_input(0, 1, 3, 2, bits, a);
    protocol.prepare_joint_input(2, 3, 1, 0, bits, b);
    P.send_receive_all(protocol.channels, protocol.send_os, protocol.receive_os);
    protocol.finalize_joint_input(0, 1, 3, 2, a);
    protocol.finalize_joint_input(2, 3, 1, 0, b);

    auto results = protocol.results;
    protocol.init_mul();
    for (int i = 0; i < batch_size; i++)
        protocol.prepare_mul(a[i].res, b[i].res);
    protocol.exchange();
    typedef typename T::clear clear;
    clear two = clear(1) + clear(1);
    for (int i = 0; i < batch_size; i++)
        this->bits.push_back(
                a[i].res + b[i].res - two * protocol.finalize_mul());
}

#endif /* PROTOCOLS_REP4PREP_HPP_ */
