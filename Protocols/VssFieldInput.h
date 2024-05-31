

#ifndef PROTOCOLS_VSSFIELDINPUT_H_
#define PROTOCOLS_VSSFIELDINPUT_H_

#include "SemiInput.h"
#include "ReplicatedInput.h"
#include "Processor/Input.h"
#include "Vss.h"
#include "Machines/vss-field-party.h"

template<class T> class VssFieldMC;

/**
 * Base class for input protocols where the inputting player sends a share
 * to every other player
 */
// template<class T>
// class IndividualVssInput : public PairwiseKeyInput<T>
// {
// protected:
//     Player& P;
//     octetStreams os;
//     vector<bool> senders;

// public:
//     IndividualVssInput(SubProcessor<T>* proc, Player& P) :
//             PairwiseKeyInput<T>(proc, P), P(P), senders(P.num_players())
//     {
//         this->reset_all(P);
//     }
//     IndividualVssInput(SubProcessor<T>& proc) :
//             IndividualVssInput<T>(&proc , proc.P)
//     {
//     }

//     void reset(int player);
//     void add_sender(int player);
//     void add_other(int player, int n_bits = -1);
//     void send_mine();
//     void exchange();
//     void finalize_other(int player, T& target, octetStream& o, int n_bits = -1);

//     void start_exchange();
//     void stop_exchange();
// };

/**
 * Vector space secret sharing over field input protocol
 */
template<class T>
class VssFieldInput : public SemiInput<T>
{
    Player& P;
    friend class Vss<T>;
    vector<vector<octetStream>> os;
    vector<bool> expect;
    int npparties;   // the number of privileged parties
    int naparties;   // the number of assistant parties 
    int ndparties;   // the number of assistant parties allowed to drop out


public:
    std::vector<typename T::open_type> inv;
    VssFieldInput(SubProcessor<T>& proc, VssFieldMC<T>&) :
            VssFieldInput<T>(&proc, proc.P)
    {
        npparties = VssFieldMachine::s().npparties;
        naparties = VssFieldMachine::s().naparties;
        ndparties = VssFieldMachine::s().ndparties;
    }

    VssFieldInput(SubProcessor<T>* proc, Player& P);

    VssFieldInput(typename T::MAC_Check& MC, Preprocessing<T>& prep, Player& P) :
            VssFieldInput(0, P)
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


#endif /* PROTOCOLS_VSSFIELDINPUT_H_ */
