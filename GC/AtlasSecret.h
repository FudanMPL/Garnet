/*
 * AtlasSecret.h
 *
 */

#ifndef GC_ATLASSECRET_H_
#define GC_ATLASSECRET_H_

#include "TinySecret.h"
#include "AtlasShare.h"

namespace GC
{

class AtlasSecret : public VectorSecret<AtlasShare>
{
    typedef AtlasSecret This;
    typedef VectorSecret<AtlasShare> super;

public:
    typedef TinyMC<This> MC;
    typedef MC MAC_Check;
    typedef VectorProtocol<This> Protocol;
    typedef VectorInput<This> Input;
    typedef CcdPrep<This> LivePrep;

    static string type_short()
    {
        return "atlas";
    }

    static MC* new_mc(typename super::mac_key_type);

    AtlasSecret()
    {
    }

    template<class T>
    AtlasSecret(const T& other) :
            super(other)
    {
    }
};

}

#endif /* GC_ATLASSECRET_H_ */
