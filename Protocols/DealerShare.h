/*
 * DealerShare.h
 *
 */

#ifndef PROTOCOLS_DEALERSHARE_H_
#define PROTOCOLS_DEALERSHARE_H_

#include "Math/Z2k.h"
#include "SemiShare.h"

template<class T> class DealerPrep;
template<class T> class DealerInput;
template<class T> class DealerMC;
template<class T> class DirectDealerMC;
template<class T> class DealerMatrixPrep;
template<class T> class Hemi;

namespace GC
{
class DealerSecret;
}

template<class T> class Dealer;

template<class T>
class DealerShare : public SemiShare<T>
{
    typedef DealerShare This;
    typedef SemiShare<T> super;

public:
    typedef GC::DealerSecret bit_type;

    typedef DealerMC<This> MAC_Check;
    typedef DirectDealerMC<This> Direct_MC;
    typedef Hemi<This> Protocol;
    typedef DealerInput<This> Input;
    typedef DealerPrep<This> LivePrep;
    typedef ::PrivateOutput<This> PrivateOutput;

    typedef DealerMatrixPrep<This> MatrixPrep;
    typedef Dealer<This> BasicProtocol;

    static false_type dishonest_majority;
    const static bool needs_ot = false;
    const static bool symmetric = false;

    static string type_short()
    {
        return "DD" + string(1, T::type_char());
    }

    static bool real_shares(const Player& P)
    {
        return P.my_num() != P.num_players() - 1;
    }

    static This constant(const T& other, int my_num,
            const typename super::mac_key_type& = {}, int = -1)
    {
        if (my_num == 1)
            return other;
        else
            return {};
    }

    DealerShare()
    {
    }

    template<class U>
    DealerShare(const U& other) : super(other)
    {
    }
};

template<int K>
using DealerRingShare = DealerShare<SignedZ2<K>>;

template<class T>
false_type DealerShare<T>::dishonest_majority;

#endif /* PROTOCOLS_DEALERSHARE_H_ */
