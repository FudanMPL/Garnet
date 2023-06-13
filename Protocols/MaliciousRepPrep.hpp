/*
 * MaliciousRepPrep.cpp
 *
 */

#ifndef PROTOCOLS_MALICIOUSREPPREP_HPP_
#define PROTOCOLS_MALICIOUSREPPREP_HPP_

#include "MaliciousRepPrep.h"
#include "Tools/Subroutines.h"
#include "Processor/OnlineOptions.h"

#include "mac_key.hpp"

template<class T>
MaliciousBitOnlyRepPrep<T>::MaliciousBitOnlyRepPrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage),
        honest_prep(0, honest_usage), honest_proc(0)
{
    this->proc = proc;
}

template<class T>
MaliciousRepPrep<T>::MaliciousRepPrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage), MaliciousBitOnlyRepPrep<T>(proc, usage)
{
}

template<class T>
MaliciousRepPrep<T>::MaliciousRepPrep(DataPositions& usage, int) :
        MaliciousRepPrep<T>(0, usage)
{
}

template<class T>
template<class V>
MaliciousRepPrep<T>::MaliciousRepPrep(DataPositions& usage,
        GC::ShareThread<V>&, int) :
        MaliciousRepPrep<T>(0, usage)
{
}

template<class T>
MaliciousRepPrepWithBits<T>::MaliciousRepPrepWithBits(SubProcessor<T>* proc,
        DataPositions& usage) :
        BufferPrep<T>(usage), MaliciousRepPrep<T>(proc, usage),
        BitPrep<T>(proc, usage), RingPrep<T>(proc, usage),
        MaliciousDabitOnlyPrep<T>(proc, usage),
        MaliciousRingPrep<T>(proc, usage)
{
}

template<class U>
MaliciousBitOnlyRepPrep<U>::~MaliciousBitOnlyRepPrep()
{
    if (honest_proc)
        delete honest_proc;
}

template<class T>
void MaliciousBitOnlyRepPrep<T>::set_protocol(typename T::Protocol& protocol)
{
    init_honest(protocol.P);
}

template<class T>
void MaliciousBitOnlyRepPrep<T>::init_honest(Player& P)
{
    if (not honest_proc)
        honest_proc = new SubProcessor<typename T::Honest>(honest_mc,
                honest_prep, P);
}

template<class T>
void MaliciousRepPrep<T>::buffer_triples()
{
    check_field_size<typename T::open_type>();
    auto& triples = this->triples;
    auto buffer_size = this->buffer_size;
    auto& honest_proc = this->honest_proc;
    assert(honest_proc != 0);
    Player& P = honest_proc->P;
    vector<array<T, 5>> check_triples;
    check_triples.reserve(buffer_size);
    auto& honest_prot = honest_proc->protocol;
    honest_prot.init_mul();
    for (int i = 0; i < buffer_size; i++)
    {
        check_triples.push_back({});
        auto& tuple = check_triples.back();
        for (int j = 0; j < 3; j++)
            tuple[j] = honest_prot.get_random();
        honest_prot.prepare_mul(tuple[0], tuple[2]);
        honest_prot.prepare_mul(tuple[1], tuple[2]);
    }
    honest_prot.exchange();
    for (int i = 0; i < buffer_size; i++)
        for (int j = 3; j < 5; j++)
            check_triples[i][j] = honest_prot.finalize_mul();
    sacrifice<T, typename T::random_type>(check_triples, P);
    for (auto& tuple : check_triples)
        triples.push_back({{tuple[0], tuple[2], tuple[3]}});
}

template<class T, class U>
void sacrifice(const vector<array<T, 5>>& check_triples, Player& P)
{
    check_field_size<U>();
    vector<T> masked, checks;
    vector <typename T::open_type> opened;
    typename T::MAC_Check MC;
    int buffer_size = check_triples.size();
    auto t = Create_Random<U>(P);
    masked.reserve(buffer_size);
    for (int i = 0; i < buffer_size; i++)
    {
        const T& a = check_triples[i][0];
        const T& a_prime = check_triples[i][1];
        masked.push_back(T::Mul(a, t) - a_prime);
    }
    MC.POpen(opened, masked, P);
    checks.reserve(buffer_size);
    for (int i = 0; i < buffer_size; i++)
    {
        const T& b = check_triples[i][2];
        const T& c = check_triples[i][3];
        const T& c_prime = check_triples[i][4];
        typename T::open_type& rho = opened[i];
        checks.push_back(T::Mul(c, t) - c_prime - rho * b);
    }
    MC.CheckFor(0, checks, P);
    MC.Check(P);
}

template<class T>
void MaliciousRepPrep<T>::buffer_squares()
{
    vector<T> masked;
    vector<T> checks;
    vector<typename T::open_type> opened;
    vector<array<T, 2>> check_squares;
    auto& squares = this->squares;
    auto buffer_size = this->buffer_size;
    auto& honest_prep = this->honest_prep;
    auto& honest_proc = this->honest_proc;
    auto& MC = this->MC;
    assert(honest_proc);
    Player& P = honest_proc->P;
    squares.clear();
    honest_prep.buffer_size = buffer_size;
    for (int i = 0; i < buffer_size; i++)
    {
        T a, b;
        T f, h;
        honest_prep.get_two(DATA_SQUARE, a, b);
        honest_prep.get_two(DATA_SQUARE, f, h);
        squares.push_back({{a, b}});
        check_squares.push_back({{f, h}});
    }
    auto t = Create_Random<typename T::clear>(P);
    for (int i = 0; i < buffer_size; i++)
    {
        T& a = squares[i][0];
        T& f = check_squares[i][0];
        masked.push_back(a * t - f);
    }
    MC.POpen(opened, masked, P);
    typename T::clear t2 = t * t;
    for (int i = 0; i < buffer_size; i++)
    {
        T& a = squares[i][0];
        T& b = squares[i][1];
        T& f = check_squares[i][0];
        T& h = check_squares[i][1];
        auto& rho = opened[i];
        checks.push_back(t2 * b - h - rho * (t * a + f));
    }
    MC.CheckFor(0, checks, P);
}

template<class T>
void MaliciousBitOnlyRepPrep<T>::buffer_bits()
{
    vector<T> masked;
    vector<T> checks;
    vector<typename T::open_type> opened;
    vector<array<T, 2>> check_squares;
    auto& bits = this->bits;
    auto buffer_size = this->buffer_size;
    assert(honest_proc);
    Player& P = honest_proc->P;
    honest_prep.buffer_size = buffer_size;
    bits.clear();
    for (int i = 0; i < buffer_size; i++)
    {
        T a, f, h;
        honest_prep.get_one(DATA_BIT, a);
        honest_prep.get_two(DATA_SQUARE, f, h);
        bits.push_back(a);
        check_squares.push_back({{f, h}});
#ifdef DEBUG_BIT_SACRIFICE
        typename T::MAC_Check MC;
        if (not (MC.open(a, P).is_zero() or MC.open(a, P).is_one()))
        {
            cout << MC.open(a, P) << endl;
            throw exception();
        }
        assert(MC.open(f, P) * MC.open(f, P) == MC.open(h, P));
#endif
    }
    auto t = Create_Random<typename T::clear>(P);
    for (int i = 0; i < buffer_size; i++)
    {
        T& a = bits[i];
        T& f = check_squares[i][0];
        masked.push_back(t * a - f);
    }
    MC.POpen(opened, masked, P);
    typename T::clear t2 = t * t;
    for (int i = 0; i < buffer_size; i++)
    {
        T& a = bits[i];
        T& f = check_squares[i][0];
        T& h = check_squares[i][1];
        auto& rho = opened[i];
        checks.push_back(t2 * a - h - rho * (t * a + f));
    }
    MC.CheckFor(0, checks, P);
}

template<class T>
void MaliciousRepPrep<T>::buffer_inputs(int player)
{
    auto proc = this->proc;
    assert(proc);
    this->buffer_inputs_as_usual(player, proc);
}

#endif
