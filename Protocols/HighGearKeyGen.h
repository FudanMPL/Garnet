/*
 * HighGearKeyGen.h
 *
 */

#ifndef PROTOCOLS_HIGHGEARKEYGEN_H_
#define PROTOCOLS_HIGHGEARKEYGEN_H_

#include "LowGearKeyGen.h"

#include <deque>
using namespace std;

template<class T, class U>
class KeyGenBitFactory
{
    U& keygen;
    deque<T>& buffer;

public:
    KeyGenBitFactory(U& keygen, deque<T>& buffer) :
            keygen(keygen), buffer(buffer)
    {
    }

    T get_bit()
    {
        if (buffer.empty())
            keygen.buffer_mabits();
        auto res = buffer.front();
        buffer.pop_front();
        return res;
    }
};

/**
 * Somewhat homomorphic encryption key generation using MASCOT
 */
template<int L, int M>
class HighGearKeyGen
{
public:
    typedef KeyGenProtocol<1, -1> Proto0;
    typedef KeyGenProtocol<2, -1> Proto1;

    typedef typename Proto0::share_type share_type0;
    typedef typename Proto1::share_type share_type1;

    typedef typename share_type0::open_type open_type0;
    typedef typename share_type1::open_type open_type1;

    typedef ShareVector<share_type0> vector_type0;
    typedef ShareVector<share_type1> vector_type1;

    typedef typename share_type0::bit_type BT;

    Player& P;
    const FHE_Params& params;

    Proto0 proto0;
    Proto1 proto1;

    deque<share_type0> bits0;
    deque<share_type1> bits1;

    HighGearKeyGen(Player& P, const FHE_Params& params);

    void buffer_mabits();

    template<class FD>
    void run(PartSetup<FD>& setup, MachineBase& machine);
};

#endif /* PROTOCOLS_HIGHGEARKEYGEN_H_ */
