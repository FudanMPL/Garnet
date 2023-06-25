/*
 * SpdzWiseMC.h
 *
 */

#ifndef PROTOCOLS_SPDZWISEMC_H_
#define PROTOCOLS_SPDZWISEMC_H_

#include "MaliciousRepMC.h"

template<class T>
class SpdzWiseMC : public MAC_Check_Base<T>
{
    vector<typename T::open_part_type> shares;

    void get_shares(const vector<T>& S)
    {
        shares.clear();
        for (auto& share : S)
            shares.push_back(share.get_share());
    }

public:
    typename T::open_part_type::MAC_Check inner_MC;

    SpdzWiseMC(typename T::mac_key_type mac_key, int = 0, int = 0, int = 0) :
            MAC_Check_Base<T>(mac_key)
    {
    }
    SpdzWiseMC(typename T::mac_key_type mac_key, Names&, int) :
            MAC_Check_Base<T>(mac_key)
    {
    }

    void init_open(const Player& P, int n = 0)
    {
        inner_MC.init_open(P, n);
    }
    void prepare_open(const T& secret, int = -1)
    {
        inner_MC.prepare_open(secret.get_share());
    }
    void exchange(const Player& P)
    {
        inner_MC.exchange(P);
    }
    typename T::open_type finalize_raw()
    {
        return inner_MC.finalize_raw();
    }
    void Check(const Player& P)
    {
        inner_MC.Check(P);
    }
    void CheckFor(const typename T::open_type& value, const vector<T>& S,
            const Player& P)
    {
        get_shares(S);
        inner_MC.CheckFor(value, shares, P);
    }
};

#endif /* PROTOCOLS_SPDZWISEMC_H_ */
