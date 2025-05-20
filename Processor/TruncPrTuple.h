/*
 * TruncPrTuple.h
 *
 */

#ifndef PROCESSOR_TRUNCPRTUPLE_H_
#define PROCESSOR_TRUNCPRTUPLE_H_

#include <vector>
#include <assert.h>
using namespace std;

#include "OnlineOptions.h"

template<class T>
class TruncPrTuple
{
public:
    const static int n = 4;

    int dest_base;
    int source_base;
    int k;
    int m;
    int n_shift;

    TruncPrTuple(const vector<int>& regs, size_t base) :
            TruncPrTuple(regs.begin() + base)
    {
    }

    TruncPrTuple(vector<int>::const_iterator it)
    {
        dest_base = *it++;
        source_base = *it++;
        k = *it++;
        m = *it++;
        n_shift = T::N_BITS - 1 - k;
        assert(m < k);
        assert(0 < k);
        assert(m < T::n_bits());
    }

    T upper(T mask)
    {
        return (mask << (n_shift + 1)) >> (n_shift + m + 1);
    }

    T msb(T mask)
    {
        return (mask << (n_shift)) >> (T::N_BITS - 1);
    }

};

template<class T>
class TruncPrTupleWithGap : public TruncPrTuple<T>
{
public:
    TruncPrTupleWithGap(const vector<int>& regs, size_t base) :
            TruncPrTupleWithGap<T>(regs.begin() + base)
    {
    }

    TruncPrTupleWithGap(vector<int>::const_iterator it) :
            TruncPrTuple<T>(it)
    {
        if (T::prime_field and small_gap())
            throw runtime_error("domain too small for chosen truncation error");
    }

    T upper(T mask)
    {
        if (big_gap())
            return mask >> this->m;
        else
            return TruncPrTuple<T>::upper(mask);
    }

    T msb(T mask)
    {
        assert(not big_gap());
        return TruncPrTuple<T>::msb(mask);
    }

    bool big_gap()
    {
        return this->k <= T::n_bits() - OnlineOptions::singleton.trunc_error;
    }

    bool small_gap()
    {
        return not big_gap();
    }
};

#endif /* PROCESSOR_TRUNCPRTUPLE_H_ */
