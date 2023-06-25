/*
 * BitPrepFiles.h
 *
 */

#ifndef GC_BITPREPFILES_H_
#define GC_BITPREPFILES_H_

#include "ShiftableTripleBuffer.h"
#include "Processor/Data_Files.h"

namespace GC
{

template<class T>
class BitPrepFiles : public ShiftableTripleBuffer<T>, public Sub_Data_Files<T>
{
public:
    BitPrepFiles(const Names& N, const string& prep_data_dir,
            DataPositions& usage, int thread_num = -1) :
            Sub_Data_Files<T>(N, prep_data_dir, usage, thread_num)
    {
    }

    array<T, 3> get_triple_no_count(int n_bits)
    {
        return ShiftableTripleBuffer<T>::get_triple_no_count(n_bits);
    }

    void get(Dtype type, T* data)
    {
        Sub_Data_Files<T>::get(type, data);
    }
};

}

#endif /* GC_BITPREPFILES_H_ */
