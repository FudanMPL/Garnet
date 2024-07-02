/*
 * VssInput.h
 *
 */

#ifndef PROTOCOLS_VSSINPUT_H_
#define PROTOCOLS_VSSINPUT_H_

#include "ReplicatedInput.h"
#include "SemiInput.h"


template<class T> class VssMC;



/**
 * Vector space secret sharing input protocol
 */
template<class T>
class VssInput : public SemiInput<T>
{
    Player& P;
    vector<vector<octetStream>> os;
    vector<bool> expect;
public:
    // std::vector<std::vector<int>> public_matrix;
    // std::vector<int> inv;
    VssInput(SubProcessor<T>& proc, VssMC<T>&) :
            VssInput(&proc, proc.P)
    {
    }

    VssInput(SubProcessor<T>* proc, Player& P);

    VssInput(typename T::MAC_Check& MC, Preprocessing<T>& prep, Player& P) :
            VssInput(0, P)
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


#endif /* PROTOCOLS_VSSINPUT_H_ */