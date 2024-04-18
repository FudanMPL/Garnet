/*
 * MaliciousCcdSecret.h
 *
 */

#ifndef GC_MALICIOUSCCDSECRET_H_
#define GC_MALICIOUSCCDSECRET_H_

#include "CcdSecret.h"
#include "MaliciousCcdShare.h"

namespace GC
{

template<class T> class VectorInput;

template<class T>
class MaliciousCcdSecret : public VectorSecret<MaliciousCcdShare<T>>
{
    typedef MaliciousCcdSecret This;
    typedef VectorSecret<MaliciousCcdShare<T>> super;

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

    MaliciousCcdSecret()
    {
    }

    MaliciousCcdSecret(const typename This::part_type& other) :
            super(other)
    {
    }

    MaliciousCcdSecret(const typename super::super& other) :
            super(other)
    {
    }

    MaliciousCcdSecret(const typename This::part_type::super& other) :
            super(other)
    {
    }

    MaliciousCcdSecret(const GC::Clear& other) :
            super(other)
    {
    }
};

} /* namespace GC */

#endif /* GC_MALICIOUSCCDSECRET_H_ */
