/*
 * SemiInput.h
 *
 */

#ifndef PROTOCOLS_SEMIINPUT_H_
#define PROTOCOLS_SEMIINPUT_H_

#include "ReplicatedInput.h"

template<class T> class SemiMC;

template<class T>
class PairwiseKeyInput : public PrepLessInput<T>
{
protected:
    vector<SeededPRNG> send_prngs;
    vector<PRNG> recv_prngs;

public:
    PairwiseKeyInput(SubProcessor<T>* proc, PlayerBase& P);
};

/**
 * Additive secret sharing input protocol
 */
template<class T>
class SemiInput : public PairwiseKeyInput<T>
{
    PlayerBase& P;

public:
    SemiInput(SubProcessor<T>& proc, SemiMC<T>&) :
            SemiInput(&proc, proc.P)
    {
    }

    SemiInput(SubProcessor<T>* proc, PlayerBase& P);

    SemiInput(typename T::MAC_Check& MC, Preprocessing<T>& prep, Player& P) :
            SemiInput(0, P)
    {
        (void) MC, (void) prep;
    }

    void reset(int player);
    void add_mine(const typename T::clear& input, int n_bits = -1);
    void add_other(int player, int n_bits = -1);
    void exchange();
    void finalize_other(int player, T& target, octetStream& o, int n_bits = -1);
    T finalize_mine();
};

#endif /* PROTOCOLS_SEMIINPUT_H_ */
