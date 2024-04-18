/*
 * NoProtocol.h
 *
 */

#ifndef PROCESSOR_NOPROTOCOL_H_
#define PROCESSOR_NOPROTOCOL_H_

#include "Protocols/Replicated.h"
#include "Protocols/MAC_Check_Base.h"
#include "Processor/Input.h"

// opening facility
template<class T>
class NoOutput : public MAC_Check_Base<T>
{
public:
    NoOutput(const typename T::mac_key_type& mac_key, int = 0, int = 0) :
            MAC_Check_Base<T>(mac_key)
    {
    }

    // open shares in this->shares and put clear values in this->values
    void exchange(const Player&)
    {
        throw runtime_error("no opening");
    }
};

// multiplication protocol
template<class T>
class NoProtocol : public ProtocolBase<T>
{
public:
    Player& P;

    static int get_n_relevant_players()
    {
        throw runtime_error("no number of relevant players");
        return -1;
    }

    NoProtocol(Player& P) :
        P(P)
    {
    }

    // prepare next round of multiplications
    void init_mul()
    {
    }

    // schedule multiplication
    void prepare_mul(const T&, const T&, int = -1)
    {
        throw runtime_error("no multiplication preparation");
    }

    // execute protocol
    void exchange()
    {
        throw runtime_error("no multiplication exchange");
    }

    // return next product
    T finalize_mul(int = -1)
    {
        throw runtime_error("no multiplication finalization");
    }
};

// private input facility
template<class T>
class NoInput : public InputBase<T>
{
public:
    NoInput(SubProcessor<T>&, typename T::MAC_Check&)
    {
    }

    // prepare next round of inputs from specific party
    void reset(int)
    {
    }

    // schedule private input from me
    void add_mine(const typename T::open_type&, int = -1)
    {
        throw runtime_error("no input from me");
    }

    // schedule private from someone else
    void add_other(int, int = -1)
    {
        throw runtime_error("no input from others");
    }

    // send my inputs
    void send_mine()
    {
        throw runtime_error("no sending of my inputs");
    }

    // return share corresponding to my next input
    T finalize_mine()
    {
        throw runtime_error("no finalizing for my inputs");
    }

    // construct share corresponding to someone else's input
    void finalize_other(int, T&, octetStream&, int = -1)
    {
        throw runtime_error("no finalizing of someone else's input");
    }
};

#endif /* PROCESSOR_NOPROTOCOL_H_ */
