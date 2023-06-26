/*
 * CcdSecret.h
 *
 */

#ifndef GC_CCDSECRET_H_
#define GC_CCDSECRET_H_

#include "TinySecret.h"
#include "CcdShare.h"

namespace GC
{

template<class T> class TinyMC;
template<class T> class VectorProtocol;
template<class T> class VectorInput;
template<class T> class CcdPrep;

template<class T>
class CcdSecret : public VectorSecret<CcdShare<T>>
{
    typedef CcdSecret This;
    typedef VectorSecret<CcdShare<T>> super;

public:
    typedef TinyMC<This> MC;
    typedef MC MAC_Check;
    typedef VectorProtocol<This> Protocol;
    typedef CcdPrep<This> LivePrep;
    typedef VectorInput<This> Input;

    typedef typename This::part_type check_type;

    static string type_short()
    {
        return "CCD";
    }

    static MC* new_mc(typename super::mac_key_type mac_key)
    {
        return new MC(mac_key);
    }

    CcdSecret()
    {
    }

    CcdSecret(const typename This::part_type& other) :
            super(other)
    {
    }

    CcdSecret(const typename super::super& other) :
            super(other)
    {
    }

    CcdSecret(const typename This::part_type::super& other) :
            super(other)
    {
    }

    CcdSecret(const GC::Clear& other) :
            super(other)
    {
    }
};

} /* namespace GC */

#endif /* GC_CCDSECRET_H_ */
