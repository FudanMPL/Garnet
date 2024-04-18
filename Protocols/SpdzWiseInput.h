/*
 * SpdzWiseInput.h
 *
 */

#ifndef PROTOCOLS_SPDZWISEINPUT_H_
#define PROTOCOLS_SPDZWISEINPUT_H_

#include "ReplicatedInput.h"

/**
 * Honest-majority input protocol with MAC
 */
template<class T>
class SpdzWiseInput : public InputBase<T>
{
    Player& P;

    typename T::part_type::Input part_input;
    typename T::part_type::Honest::Protocol honest_mult;

    typename T::Protocol checker;
    SubProcessor<T>* proc;

    typename T::mac_key_type mac_key;

    vector<int> counters;
    vector<PointerVector<T>> shares;

public:
    SpdzWiseInput(SubProcessor<T>& proc, Player& P);
    SpdzWiseInput(SubProcessor<T>* proc, Player& P);
    SpdzWiseInput(SubProcessor<T>& proc, typename T::MAC_Check& MC);
    ~SpdzWiseInput();

    void reset(int player);
    void add_mine(const typename T::open_type& input, int n_bits = -1);
    void add_other(int player, int n_bits = -1);
    void exchange();
    T finalize(int player, int n_bits = -1);
};

#endif /* PROTOCOLS_SPDZWISEINPUT_H_ */
