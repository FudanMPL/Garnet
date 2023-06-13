/*
 * ShiftableTripleBuffer.h
 *
 */

#ifndef GC_SHIFTABLETRIPLEBUFFER_H_
#define GC_SHIFTABLETRIPLEBUFFER_H_

#include "Math/FixedVec.h"

#include <assert.h>

namespace GC
{

template<class T>
class ShiftableTripleBuffer
{
    FixedVec<T, 3> triple_buffer;
    int buffer_left;

    virtual void get(Dtype type, T* data) = 0;

public:
    ShiftableTripleBuffer() :
            buffer_left(0)
    {
    }

    virtual ~ShiftableTripleBuffer() {}

    array<T, 3> get_triple_no_count(int n_bits)
    {
        int max_n_bits = T::default_length;
        assert(n_bits <= max_n_bits);
        assert(n_bits > 0);
        array<T, 3> res;

        if (n_bits <= buffer_left)
        {
            res = triple_buffer.mask(n_bits).get();
            triple_buffer >>= n_bits;
            buffer_left -= n_bits;
        }
        else
        {
            get(DATA_TRIPLE, res.data());
            FixedVec<T, 3> tmp = res;
            res = tmp.mask(n_bits).get();
            triple_buffer += (tmp >> n_bits) << buffer_left;
            buffer_left += max_n_bits - n_bits;
        }

        return res;
    }
};

} /* namespace GC */

#endif /* GC_SHIFTABLETRIPLEBUFFER_H_ */
