/*
 * MaliciousRepSecret.h
 *
 */

#ifndef GC_MALICIOUSREPSECRET_H_
#define GC_MALICIOUSREPSECRET_H_

#include "ShareSecret.h"
#include "Machine.h"
#include "ThreadMaster.h"
#include "Protocols/Beaver.h"
#include "Protocols/MaliciousRepMC.h"
#include "Processor/DummyProtocol.h"

template<class T> class MaliciousRepMC;

namespace GC
{

template<class T> class ShareThread;
template<class T> class RepPrep;

class SmallMalRepSecret : public FixedVec<BitVec_<unsigned char>, 2>
{
    typedef FixedVec<BitVec_<unsigned char>, 2> super;
    typedef SmallMalRepSecret This;

public:
    typedef MaliciousRepMC<This> MC;
    typedef BitVec_<unsigned char> open_type;
    typedef open_type clear;
    typedef NoValue mac_key_type;

    static MC* new_mc(mac_key_type)
    {
        return new HashMaliciousRepMC<This>;
    }

    SmallMalRepSecret()
    {
    }
    template<class T>
    SmallMalRepSecret(const T& other) :
            super(other)
    {
    }

    This lsb() const
    {
        return *this & 1;
    }
};

template<class U>
class MalRepSecretBase : public ReplicatedSecret<U>
{
    typedef ReplicatedSecret<U> super;

public:
    typedef Memory<U> DynamicMemory;

    typedef MaliciousRepMC<U> MC;
    typedef MC MAC_Check;

    typedef ReplicatedInput<U> Input;
    typedef RepPrep<U> LivePrep;

    typedef U part_type;
    typedef U whole_type;

    static const bool expensive_triples = true;

    static MC* new_mc(typename super::mac_key_type)
    {
        try
        {
            if (ThreadMaster<U>::s().machine.more_comm_less_comp)
                return new CommMaliciousRepMC<U>;
        }
        catch(no_singleton& e)
        {
        }
        return new HashMaliciousRepMC<U>;
    }

    MalRepSecretBase() {}
    template<class T>
    MalRepSecretBase(const T& other) : super(other) {}
};

class MaliciousRepSecret : public MalRepSecretBase<MaliciousRepSecret>
{
    typedef MaliciousRepSecret This;
    typedef MalRepSecretBase<This> super;

public:
    typedef Beaver<MaliciousRepSecret> Protocol;

    typedef SmallMalRepSecret small_type;

    MaliciousRepSecret() {}
    template<class T>
    MaliciousRepSecret(const T& other) : super(other) {}
};

}

#endif /* GC_MALICIOUSREPSECRET_H_ */
