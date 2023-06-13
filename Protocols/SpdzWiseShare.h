/*
 * SpdzWiseShare.h
 *
 */

#ifndef PROTOCOLS_SPDZWISESHARE_H_
#define PROTOCOLS_SPDZWISESHARE_H_

#include "Share.h"
#include "SpdzWise.h"
#include "Processor/DummyProtocol.h"

template<class T> class NoLivePrep;
template<class T> class NotImplementedInput;
template<class T> class SpdzWiseMC;
template<class T> class SpdzWisePrep;
template<class T> class SpdzWiseInput;

namespace GC
{
class MaliciousRepSecret;
}

template<class T>
class SpdzWiseShare : public Share_<T, T>
{
    typedef Share_<T, T> super;

public:
    typedef T part_type;
    typedef T open_part_type;
    typedef typename T::clear clear;
    typedef typename T::open_type open_type;

    typedef SpdzWiseMC<SpdzWiseShare> MAC_Check;
    typedef MAC_Check Direct_MC;

    typedef SpdzWise<SpdzWiseShare> Protocol;
    typedef SpdzWisePrep<SpdzWiseShare> LivePrep;
    typedef SpdzWiseInput<SpdzWiseShare> Input;
    typedef ::PrivateOutput<SpdzWiseShare> PrivateOutput;

    typedef typename T::bit_type bit_type;

    static const bool expensive = true;

    static string type_short()
    {
        return "SY" + T::type_short();
    }

    static string type_string()
    {
        return "SPDZ-wise " + T::type_string();
    }

    static void read_or_generate_mac_key(string directory, Player& P, T& mac_key);

    static open_type get_rec_factor(int i, int n)
    {
        return T::get_rec_factor(i, n);
    }

    static void specification(octetStream& os)
    {
        T::specification(os);
    }

    SpdzWiseShare()
    {
    }

    SpdzWiseShare(const super& other) :
            super(other)
    {
    }

    SpdzWiseShare(const T& share, const T& mac) :
            super(share, mac)
    {
    }

    void pack(octetStream& os, bool full = true) const;
    void pack(octetStream& os, open_type factor) const;
};

template<class T> class MaliciousRep3Share;
template<class T>
using SpdzWiseRepFieldShare = SpdzWiseShare<MaliciousRep3Share<T>>;

#endif /* PROTOCOLS_SPDZWISESHARE_H_ */
