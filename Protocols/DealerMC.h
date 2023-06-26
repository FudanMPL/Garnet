/*
 * DealerMC.h
 *
 */

#ifndef PROTOCOLS_DEALERMC_H_
#define PROTOCOLS_DEALERMC_H_

#include "MAC_Check_Base.h"
#include "Networking/AllButLastPlayer.h"

template<class T>
class DealerMC : public MAC_Check_Base<T>
{
    typedef SemiMC<SemiShare<typename T::clear>> internal_type;
    internal_type& internal;
    AllButLastPlayer* sub_player;

public:
    DealerMC(typename T::mac_key_type = {}, int = 0, int = 0);
    DealerMC(internal_type& internal);
    ~DealerMC();

    void init_open(const Player& P, int n = 0);
    void prepare_open(const T& secret, int n_bits = -1);
    void exchange(const Player& P);
    typename T::open_type finalize_raw();
    array<typename T::open_type*, 2> finalize_several(int n);

    DealerMC& get_part_MC()
    {
        return *this;
    }
};

template<class T>
class DirectDealerMC : public DealerMC<T>
{
public:
    DirectDealerMC(typename T::mac_key_type = {});
};

#endif /* PROTOCOLS_DEALERMC_H_ */
