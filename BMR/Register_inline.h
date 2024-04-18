/*
 * Register_inline.h
 *
 */

#ifndef BMR_REGISTER_INLINE_H_
#define BMR_REGISTER_INLINE_H_

#include "CommonParty.h"
#include "Party.h"

inline Register::Register() :
        garbled_entry(CommonParty::s().get_n_parties()), external(NO_SIGNAL),
        mask(NO_SIGNAL), keys(CommonParty::s().get_n_parties())
{
}

#endif /* BMR_REGISTER_INLINE_H_ */
