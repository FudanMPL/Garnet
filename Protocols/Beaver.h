/*
 * Beaver.h
 *
 */

#ifndef PROTOCOLS_BEAVER_H_
#define PROTOCOLS_BEAVER_H_

#include <vector>
#include <array>
using namespace std;

#include "Replicated.h"
#include "Processor/Data_Files.h"

template<class T> class SubProcessor;
template<class T> class MAC_Check_Base;
class Player;

/**
 * Beaver multiplication
 */
template<class T>
class Beaver : public ProtocolBase<T>
{
protected:
    vector<T> shares;
    vector<typename T::open_type> opened;
    vector<array<T, 3>> triples;
    vector<int> lengths;
    typename vector<typename T::open_type>::iterator it;
    typename vector<array<T, 3>>::iterator triple;
    Preprocessing<T>* prep;
    typename T::MAC_Check* MC;

public:
    static const bool uses_triples = true;

    Player& P;

    Beaver(Player& P) : prep(0), MC(0), P(P) {}

    typename T::Protocol branch();

    void init(Preprocessing<T>& prep, typename T::MAC_Check& MC);

    void init_mul();
    void prepare_mul(const T& x, const T& y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);

    void check();

    void start_exchange();
    void stop_exchange();

    int get_n_relevant_players() { return 1 + T::threshold(P.num_players()); }
};

#endif /* PROTOCOLS_BEAVER_H_ */
