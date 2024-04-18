/*
 * PersonalPrep.cpp
 *
 */

#ifndef GC_PERSONALPREP_HPP_
#define GC_PERSONALPREP_HPP_

#include "PersonalPrep.h"

#include "Protocols/ShuffleSacrifice.hpp"

namespace GC
{

template<class T>
PersonalPrep<T>::PersonalPrep(DataPositions& usage, int input_player) :
        BufferPrep<T>(usage), input_player(input_player)
{
    assert((input_player >= 0) or (input_player == SECURE));
}

template<class T>
void PersonalPrep<T>::buffer_personal_triples()
{
    buffer_personal_triples(OnlineOptions::singleton.batch_size);
}

template<class T>
void PersonalPrep<T>::buffer_personal_triples(size_t batch_size, ThreadQueues* queues)
{
    TripleShuffleSacrifice<T> sacri;
    batch_size = max(batch_size, (size_t)sacri.minimum_n_outputs()) + sacri.C;
    vector<array<T, 3>> triples(batch_size);

    if (queues)
    {
        PersonalTripleJob job(&triples, input_player);
        int start = queues->distribute(job, batch_size);
        buffer_personal_triples(triples, start, batch_size);
        if (start)
            queues->wrap_up(job);
    }
    else
        buffer_personal_triples(triples, 0, batch_size);

    auto &party = ShareThread<typename T::whole_type>::s();
    assert(party.P != 0);
    assert(party.MC != 0);
    auto& MC = party.MC->get_part_MC();
    auto& P = *party.P;
    GlobalPRNG G(P);
    vector<T> shares;
    for (int i = 0; i < sacri.C; i++)
    {
        int challenge = G.get_uint(triples.size());
        for (auto& x : triples[challenge])
            shares.push_back(x);
        triples.erase(triples.begin() + challenge);
    }
    PointerVector<typename T::open_type> opened;
    MC.POpen(opened, shares, P);
    for (int i = 0; i < sacri.C; i++)
    {
        array<typename T::open_type, 3> triple({{opened.next(), opened.next(),
            opened.next()}});
        if (triple[0] * triple[1] != triple[2])
        {
            cout << triple[2] << " != " << triple[0] * triple[1] << " = "
                    << triple[0] << " * " << triple[1] << endl;
            throw runtime_error("personal triple incorrect");
        }
    }

    this->triples.insert(this->triples.end(), triples.begin(), triples.end());
}

template<class T>
void PersonalPrep<T>::buffer_personal_triples(vector<array<T, 3>>& triples,
        size_t begin, size_t end)
{
#ifdef VERBOSE_EDA
    fprintf(stderr, "personal triples %zu to %zu\n", begin, end);
    RunningTimer timer;
#endif
    auto& party = ShareThread<typename T::whole_type>::s();
    auto& MC = party.MC->get_part_MC();
    auto& P = *party.P;
    assert(input_player < P.num_players());
    typename T::Input input(MC, *this, P);
    input.reset_all(P);
    for (size_t i = begin; i < end; i++)
    {
        typename T::open_type x[2];
        for (int j = 0; j < 2; j++)
            this->get_input(triples[i][j], x[j], input_player);
        if (P.my_num() == input_player)
            input.add_mine(x[0] * x[1], T::default_length);
        else
            input.add_other(input_player);
    }
    input.exchange();
    for (size_t i = begin; i < end; i++)
        triples[i][2] = input.finalize(input_player, T::default_length);
#ifdef VERBOSE_EDA
    fprintf(stderr, "personal triples took %f seconds\n", timer.elapsed());
#endif
}

}

#endif
