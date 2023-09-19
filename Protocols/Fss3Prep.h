/*
 * Fss3Prep.h
 *
 */

#ifndef PROTOCOLS_FSS3PREP_H_
#define PROTOCOLS_FSS3PREP_H_

#include "ReplicatedPrep.h"
#include "GC/SemiSecret.h"
#include "Tools/aes.h"
#include <vector>
#include <iostream>


template<class T>
class Fss3Prep : public virtual SemiHonestRingPrep<T>,
        public virtual ReplicatedRingPrep<T>
{
    void buffer_dabits(ThreadQueues*);
    


protected:
    virtual void get_dcf_no_count(T&a, int n_bits);
    virtual void get_dpf_no_count(T&a, int n_bits);


public:
    Fss3Prep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage), BitPrep<T>(proc, usage),
			RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage), ReplicatedRingPrep<T>(proc, usage)
    {
    }

    void buffer_bits() { this->buffer_bits_without_check(); }

    void gen_fake_dcf(int beta, int lambda);

    void gen_fake_multi_spline_dcf(SubProcessor<T> &processor, int beta, int lambda, int base, int length);
    
    void get_one_no_count(Dtype dtype, T& a)
    {
        std::cout << "jumping into Fss3Prep.h get_one_no_count" << std::endl;
        // judge the type of preprocessing values
        if (dtype ==  DATA_BIT){
            typename T::bit_type b;
            this->get_dabit_no_count(a, b);
        }
        else{
            throw not_implemented();
        }
    }
};

#endif /* PROTOCOLS_FSS3PREP_H_ */
