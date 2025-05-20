/*
 * FssShare.h
 *
 */

#ifndef PROTOCOLS_FSS3SHARE_H_
#define PROTOCOLS_FSS3SHARE_H_

#include "Rep3Share.h"
#include "Protocols/Fss.h"

template<class T> class Fss;
template<class T> class ReplicatedPrep;
template<class T> class SemiRep3Prep;
template<class T> class Fss3Prep;
template<class T> class ReplicatedRingPrep;
template<class T> class ReplicatedPO;
template<class T> class SpecificPrivateOutput;


template<class T>
class Fss3Share : public RepShare<T, 2>
{
    typedef RepShare<T, 2> super;
    typedef Fss3Share This;

public:
    typedef T clear;

    typedef Fss<This> Protocol;
    typedef ReplicatedMC<This> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef ReplicatedInput<This> Input;
    typedef ReplicatedPO<This> PO;
    typedef SpecificPrivateOutput<This> PrivateOutput;
    typedef typename conditional<T::characteristic_two,
            ReplicatedPrep<This>, Fss3Prep<This>>::type LivePrep;
    typedef ReplicatedRingPrep<This> TriplePrep;
    typedef This Honest;

    typedef This Scalar;

    typedef GC::SemiHonestRepSecret bit_type;

    const static bool needs_ot = false;
    const static bool dishonest_majority = false;
    const static bool expensive = false;
    const static bool variable_players = false;
    static const bool has_trunc_pr = true;
    static const bool malicious = false;

    static string type_short()
    {
        return "R" + string(1, clear::type_char());
    }

    static string type_string()
    {
        return "replicated " + T::type_string();
    }

    static char type_char()
    {
        return T::type_char();
    }

    static Fss3Share constant(T value, int my_num,
            typename super::mac_key_type = {})
    {
        return Fss3Share(value, my_num);
    }

    Fss3Share()
    {
    }
    template<class U>
    Fss3Share(const U& other) :
            super(other)
    {
    }
    Fss3Share(T value, int my_num, const T& alphai = {})
    {
        (void) alphai;
        Fss<Fss3Share>::assign(*this, value, my_num);
    }

    void assign(const char* buffer)
    {
        FixedVec<T, 2>::assign(buffer);
    }

    clear local_mul(const Fss3Share& other) const
    {
        auto a = (*this)[0].lazy_mul(other.lazy_sum());
        auto b = (*this)[1].lazy_mul(other[0]);
        return a.lazy_add(b);
    }
};
#endif //PROTOCOLS_FSS3SHARE_H_
