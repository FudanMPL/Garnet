/*
 * TinyMC.h
 *
 */

#ifndef GC_TINYMC_H_
#define GC_TINYMC_H_

#include "Protocols/MAC_Check_Base.h"

namespace GC
{

template<class T>
class TinyMC : public MAC_Check_Base<T>
{
    typename T::part_type::MAC_Check part_MC;
    PointerVector<int> sizes;

public:
    static void setup(Player& P)
    {
        T::part_type::MAC_Check::setup(P);
    }

    static void teardown()
    {
        T::part_type::MAC_Check::teardown();
    }

    TinyMC(typename T::mac_key_type mac_key) :
            part_MC(mac_key)
    {
        this->alphai = mac_key;
    }

    typename T::part_type::MAC_Check& get_part_MC()
    {
        return part_MC;
    }

    void init_open(const Player& P, int n)
    {
        part_MC.init_open(P);
        sizes.clear();
        sizes.reserve(n);
    }

    void prepare_open(const T& secret, int = -1)
    {
        for (auto& part : secret.get_regs())
            part_MC.prepare_open(part);
        sizes.push_back(secret.get_regs().size());
    }

    void exchange(const Player& P)
    {
        part_MC.exchange(P);
    }

    typename T::open_type finalize_raw()
    {
        int n = sizes.next();
        typename T::open_type opened = 0;
        for (int i = 0; i < n; i++)
            opened += typename T::open_type(part_MC.finalize_raw().get_bit(0)) << i;
        return opened;
    }

    void Check(const Player& P)
    {
        part_MC.Check(P);
    }
};

} /* namespace GC */

#endif /* GC_TINYMC_H_ */
