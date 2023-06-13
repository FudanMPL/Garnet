/*
 * AtlasShare.h
 *
 */

#ifndef GC_ATLASSHARE_H_
#define GC_ATLASSHARE_H_

#include "Protocols/AtlasShare.h"
#include "Protocols/ShamirMC.h"
#include "Math/Bit.h"

namespace GC
{

class AtlasSecret;

class AtlasShare : public ::AtlasShare<gf2n_<octet>>, public ShareSecret<AtlasSecret>
{
    typedef AtlasShare This;

public:
    typedef ::AtlasShare<gf2n_<octet>> super;

    typedef Atlas<This> Protocol;
    typedef ShamirMC<This> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef ShamirInput<This> Input;
    typedef ReplicatedPrep<This> LivePrep;

    typedef This small_type;

    typedef Bit clear;

    static MAC_Check* new_mc(mac_key_type)
    {
        return new MAC_Check;
    }

    AtlasShare()
    {
    }

    template<class U>
    AtlasShare(const U& other) :
            super(other)
    {
    }

    AtlasShare(const AtlasSecret& other);

    void XOR(const This& a, const This& b)
    {
        *this = a + b;
    }

    void public_input(bool input)
    {
        *this = input;
    }
};

}

#endif /* GC_ATLASSHARE_H_ */
