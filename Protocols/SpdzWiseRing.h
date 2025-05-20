/*
 * SpdzWiseRing.h
 *
 */

#ifndef PROTOCOLS_SPDZWISERING_H_
#define PROTOCOLS_SPDZWISERING_H_

#include "SpdzWise.h"
#include "PostSacrifice.h"
#include "PostSacriRepRingShare.h"

/**
 * Three-party replicated secret sharing protocol with MAC modulo a power of two
 */
template<class T>
class SpdzWiseRing : public SpdzWise<T>
{
    typedef typename T::part_type check_type;
    typedef PostSacriRepRingShare<T::LENGTH + T::SECURITY, T::SECURITY> zero_check_type;

    DataPositions zero_usage;
    MaliciousBitOnlyRepPrep<zero_check_type> zero_prep;
    typename zero_check_type::MAC_Check zero_output;
    SubProcessor<zero_check_type> zero_proc;

public:
    SpdzWiseRing(Player &P);

    void zero_check(check_type t);
};

#endif /* PROTOCOLS_SPDZWISERING_H_ */
