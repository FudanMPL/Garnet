/*
 * DealerInput.h
 *
 */

#ifndef PROTOCOLS_DEALERINPUT_H_
#define PROTOCOLS_DEALERINPUT_H_

#include "../Networking/AllButLastPlayer.h"
#include "Processor/Input.h"

template<class T>
class DealerInput : public InputBase<T>
{
    Player& P;
    octetStreams to_send, to_receive;
    SeededPRNG G;
    vector<SemiShare<typename T::clear>> shares;
    bool from_dealer;
    AllButLastPlayer sub_player;
    SemiInput<SemiShare<typename T::clear>>* internal;

public:
    DealerInput(SubProcessor<T>& proc, typename T::MAC_Check&);
    DealerInput(typename T::MAC_Check&, Preprocessing<T>&, Player& P);
    DealerInput(Player& P);
    DealerInput(SubProcessor<T>*, Player& P);
    ~DealerInput();

    bool is_dealer(int player = -1);

    void reset(int player);
    void add_mine(const typename T::open_type& input, int n_bits = -1);
    void add_other(int player, int n_bits = -1);
    void exchange();
    T finalize(int player, int n_bits = -1);
};

#endif /* PROTOCOLS_DEALERINPUT_H_ */
