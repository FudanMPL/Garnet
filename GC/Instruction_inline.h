/*
 * Instruction_inline.h
 *
 */

#ifndef GC_INSTRUCTION_INLINE_H_
#define GC_INSTRUCTION_INLINE_H_

#include "BMR/config.h"

#ifdef MAX_INLINE
#define MAYBE_INLINE inline
#else
#define MAYBE_INLINE
#endif

namespace GC {

#include "instructions.h"

template <class T>
inline bool fallback_code(const Instruction& instruction, Processor<T>& processor)
{
    (void)processor;
    cout << "Undefined instruction " << showbase << hex
            << instruction.get_opcode() << endl << dec;
    return true;
}

} /* namespace GC */

#endif /* GC_INSTRUCTION_INLINE_H_ */
