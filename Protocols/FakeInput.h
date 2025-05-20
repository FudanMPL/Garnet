/*
 * FakeProtocol.h
 *
 */

#ifndef PROTOCOLS_FAKEINPUT_H_
#define PROTOCOLS_FAKEINPUT_H_

#include "Replicated.h"
#include "Processor/Input.h"

template<class T>
class FakeInput : public InputBase<T>
{
    PointerVector<T> results;

public:
    FakeInput()
    {
    }

    FakeInput(SubProcessor<T>&, typename T::MAC_Check&)
    {
    }

    void reset(int)
    {
        results.clear();
    }

    void add_mine(const typename T::open_type& x, int = -1)
    {
        results.push_back(x);
    }

    void add_other(int, int = -1)
    {
    }

    void send_mine()
    {
    }

    T finalize_mine()
    {
        return results.next();
    }

    void finalize_other(int, T&, octetStream&, int = -1)
    {
        throw runtime_error("input from other parties not implemented");
    }
};

#endif /* PROTOCOLS_FAKEPROTOCOL_H_ */
