/*
 * PrivateOutput.h
 *
 */

#ifndef PROCESSOR_PRIVATEOUTPUT_H_
#define PROCESSOR_PRIVATEOUTPUT_H_

#include <deque>
using namespace std;

template<class T> class SubProcessor;

template<class T>
class PrivateOutput
{
    typedef typename T::open_type open_type;

    SubProcessor<T>& proc;
    typename T::MAC_Check MC;
    deque<open_type> masks;

public:
    PrivateOutput(SubProcessor<T>& proc);
    ~PrivateOutput();

    void prepare_sending(const T& source, int player);
    void exchange();
    typename T::clear finalize(int player);
};

#endif /* PROCESSOR_PRIVATEOUTPUT_H_ */
