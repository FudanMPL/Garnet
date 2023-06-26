/*
 * SpdzWise.h
 *
 */

#ifndef PROTOCOLS_SPDZWISE_H_
#define PROTOCOLS_SPDZWISE_H_

#include "Replicated.h"

template<class T> class SpdzWiseInput;

/**
 * Honest-majority protocol with MAC check
 */
template<class T>
class SpdzWise : public ProtocolBase<T>
{
    typedef typename T::part_type check_type;

    friend class SpdzWiseInput<T>;

    typename T::part_type::Honest::Protocol internal, internal2;

    typename T::mac_key_type mac_key;

    vector<T> results;

    vector<typename T::part_type> coefficients;

    void buffer_random();

    virtual void zero_check(check_type t);

public:
    Player& P;

    SpdzWise(Player& P);
    virtual ~SpdzWise();

    typename T::Protocol branch();

    void init(Preprocessing<T>&, typename T::MAC_Check& MC);

    void init_mul();
    void prepare_mul(const T& x, const T& y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);

    void init_dotprod();
    void prepare_dotprod(const T& x, const T& y);
    void next_dotprod();
    T finalize_dotprod(int length);

    void add_to_check(const T& x);
    void check();
    void maybe_check();

    int get_n_relevant_players() { return internal.get_n_relevant_players(); }

    void randoms_inst(vector<T>& S, const Instruction& instruction);
};

#endif /* PROTOCOLS_SPDZWISE_H_ */
