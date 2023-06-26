/*
 * ReplicatedInput.h
 *
 */

#ifndef PROTOCOLS_REPLICATEDINPUT_H_
#define PROTOCOLS_REPLICATEDINPUT_H_

#include "Processor/Input.h"
#include "Processor/Processor.h"
#include "Replicated.h"

/**
 * Base class for input protocols without preprocessing
 */
template <class T>
class PrepLessInput : public InputBase<T>
{
protected:
    PointerVector<T> shares;

public:
    PrepLessInput(SubProcessor<T>* proc) :
            InputBase<T>(proc ? proc->Proc : 0) {}
    virtual ~PrepLessInput() {}

    virtual void reset(int player) = 0;
    virtual void add_mine(const typename T::open_type& input,
            int n_bits = -1) = 0;
    virtual void add_other(int player, int n_bits = - 1) = 0;
    virtual void finalize_other(int player, T& target, octetStream& o,
            int n_bits = -1) = 0;

    T finalize_mine();
};

/**
 * Replicated three-party input protocol
 */
template <class T>
class ReplicatedInput : public PrepLessInput<T>
{
    SubProcessor<T>* proc;
    Player& P;
    vector<octetStream> os;
    SeededPRNG secure_prng;
    ReplicatedBase protocol;
    vector<bool> expect;

public:
    ReplicatedInput(SubProcessor<T>& proc) :
            ReplicatedInput(&proc, proc.P)
    {
    }
    ReplicatedInput(SubProcessor<T>& proc, ReplicatedMC<T>& MC) :
            ReplicatedInput(proc)
    {
        (void) MC;
    }
    ReplicatedInput(typename T::MAC_Check& MC, Preprocessing<T>& prep, Player& P) :
            ReplicatedInput(P)
    {
        (void) MC, (void) prep;
    }
    ReplicatedInput(Player& P) :
            ReplicatedInput(0, P)
    {
    }
    ReplicatedInput(SubProcessor<T>* proc, Player& P) :
            PrepLessInput<T>(proc), proc(proc), P(P), protocol(P)
    {
        assert(T::vector_length == 2);
        expect.resize(P.num_players());
        this->reset_all(P);
    }

    void reset(int player);
    void add_mine(const typename T::open_type& input, int n_bits = -1);
    void add_other(int player, int n_bits = -1);
    void send_mine();
    void exchange();
    void finalize_other(int player, T& target, octetStream& o, int n_bits = -1);
};

#endif /* PROTOCOLS_REPLICATEDINPUT_H_ */
