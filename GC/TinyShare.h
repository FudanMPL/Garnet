/*
 * TinyShare.h
 *
 */

#ifndef GC_TINYSHARE_H_
#define GC_TINYSHARE_H_

#include "ShareSecret.h"
#include "ShareParty.h"
#include "Secret.h"
#include "Protocols/Spdz2kShare.h"


namespace GC
{

template<int S> class TinySecret;
template<class T> class ShareThread;
template<class T> class TinierSharePrep;

template<int S>
class TinyShare : public Spdz2kShare<1, S>, public ShareSecret<TinySecret<S>>
{
    typedef TinyShare This;

public:
    typedef Spdz2kShare<1, S> super;

    typedef void DynamicMemory;

    typedef Beaver<This> Protocol;
    typedef MAC_Check_Z2k_<This> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef ::Input<This> Input;
    typedef TinierSharePrep<This> LivePrep;

    typedef SwitchableOutput out_type;

    typedef This small_type;

    typedef NoShare bit_type;

    static string name()
    {
        return "tiny share";
    }

    TinyShare()
    {
    }
    TinyShare(const typename super::super::super& other) :
            super(other)
    {
    }
    TinyShare(const super& other) :
            super(other)
    {
    }

    void XOR(const This& a, const This& b)
    {
        *this = a + b;
    }

    void public_input(bool input)
    {
        auto& party = this->get_party();
        *this = super::constant(input, party.P->my_num(),
                party.MC->get_alphai());
    }
};

} /* namespace GC */

#endif /* GC_TINYSHARE_H_ */
