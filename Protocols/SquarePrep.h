/*
 * SquarePrep.h
 *
 */

#ifndef PROTOCOLS_SQUAREPREP_H_
#define PROTOCOLS_SQUAREPREP_H_

#include "ReplicatedPrep.h"

template<class T, class U>
void generate_squares(vector<array<T, 2>>& squares, int n_squares,
        U* protocol);

template<class T>
class SquarePrep : public BufferPrep<T>
{
    void buffer_triples()
    {
        throw runtime_error("no triples here");
    }

    void buffer_squares()
    {
        generate_squares(this->squares, this->buffer_size,
                &this->proc->protocol);
    }

public:
    SquarePrep(DataPositions& usage) :
            BufferPrep<T>(usage)
    {
    }

    void set_protocol(typename T::Protocol&)
    {
    }
};

#endif /* PROTOCOLS_SQUAREPREP_H_ */
