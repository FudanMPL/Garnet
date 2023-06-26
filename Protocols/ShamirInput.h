/*
 * ShamirInput.h
 *
 */

#ifndef PROTOCOLS_SHAMIRINPUT_H_
#define PROTOCOLS_SHAMIRINPUT_H_

#include "Processor/Input.h"
#include "Shamir.h"
#include "SemiInput.h"
#include "Machines/ShamirMachine.h"

/**
 * Base class for input protocols where the inputting player sends a share
 * to every other player
 */
template<class T>
class IndividualInput : public PairwiseKeyInput<T>
{
protected:
    Player& P;
    octetStreams os;
    vector<bool> senders;

public:
    IndividualInput(SubProcessor<T>* proc, Player& P) :
            PairwiseKeyInput<T>(proc, P), P(P), senders(P.num_players())
    {
        this->reset_all(P);
    }
    IndividualInput(SubProcessor<T>& proc) :
            IndividualInput<T>(&proc , proc.P)
    {
    }

    void reset(int player);
    void add_sender(int player);
    void add_other(int player, int n_bits = -1);
    void send_mine();
    void exchange();
    void finalize_other(int player, T& target, octetStream& o, int n_bits = -1);

    void start_exchange();
    void stop_exchange();
};

/**
 * Shamir secret sharing input protocol
 */
template<class T>
class ShamirInput : public IndividualInput<T>
{
    friend class Shamir<T>;

    vector<vector<typename T::open_type>> reconstruction;

    vector<typename T::Scalar> randomness;

    int threshold;

    void init();

public:
    static vector<vector<typename T::open_type>> get_vandermonde(size_t t,
            size_t n);

    ShamirInput(SubProcessor<T>& proc, typename T::MAC_Check& MC) :
            ShamirInput<T>(&proc, proc.P)
    {
        (void) MC;
    }

    ShamirInput(SubProcessor<T>* proc, Player& P, int t = 0) :
            IndividualInput<T>(proc, P)
    {
        if (t > 0)
            threshold = t;
        else
            threshold = ShamirMachine::s().threshold;

        init();
    }

    ShamirInput(ShamirMC<T>&, Preprocessing<T>&, Player& P) :
            ShamirInput<T>(0, P)
    {
    }

    void add_mine(const typename T::open_type& input, int n_bits = -1);
    void finalize_other(int player, T& target, octetStream& o, int n_bits = -1);
};

#endif /* PROTOCOLS_SHAMIRINPUT_H_ */
