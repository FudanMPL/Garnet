/*
 * Clear.h
 *
 */

#ifndef GC_CLEAR_H_
#define GC_CLEAR_H_

#include "Math/Integer.h"

namespace GC
{

class Clear : public Integer
{
public:
    static string type_string() { return "clear"; }

    Clear() : Integer() {}
    Clear(long a) : Integer(a) {}
    Clear(const IntBase& x) { IntBase::operator=(x); }

    void xor_(const Clear& x, const Clear& y) { a = x.a ^ y.a; }
    void xor_(const Clear& x, long y) { a = x.a ^ y; }
};

} /* namespace GC */

#endif /* GC_CLEAR_H_ */
