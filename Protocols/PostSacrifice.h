/*
 * PostSacrifice.h
 *
 */

#ifndef PROTOCOLS_POSTSACRIFICE_H_
#define PROTOCOLS_POSTSACRIFICE_H_

#include "Protocols/Replicated.h"

/**
 * Protocol with optimistic multiplication and postponed sacrifice
 */
template<class T>
class PostSacrifice : public ProtocolBase<T>
{
    typedef typename T::prep_type prep_type;

    typename prep_type::Honest::Protocol internal;
    typename T::Honest::Protocol randomizer;

    vector<array<prep_type, 2>> operands;
    vector<prep_type> results;

public:
    Player& P;

    PostSacrifice(Player& P);
    ~PostSacrifice();

    Player& branch();

    void init_mul();
    void prepare_mul(const T& x, const T& y, int n = -1);
    void exchange() { internal.exchange(); }
    T finalize_mul(int n = -1);

    void check();

    int get_n_relevant_players() { return internal.get_n_relevant_players(); }

    virtual void randoms(T& res, int n_bits) { randomizer.randoms(res, n_bits); }
};

#endif /* PROTOCOLS_POSTSACRIFICE_H_ */
