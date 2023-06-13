/*
 * PointerVector.h
 *
 */

#ifndef TOOLS_POINTERVECTOR_H_
#define TOOLS_POINTERVECTOR_H_

#include "CheckVector.h"

template<class T>
class PointerVector : public CheckVector<T>
{
    int i;

public:
    PointerVector() : i(0) {}
    PointerVector(size_t size) : CheckVector<T>(size), i(0) {}
    PointerVector(const vector<T>& other) : CheckVector<T>(other), i(0) {}
    void clear()
    {
        vector<T>::clear();
        reset();
    }
    void reset()
    {
        i = 0;
    }
    T& next()
    {
        return (*this)[i++];
    }
    T* skip(size_t n)
    {
        i += n;
        return &*(this->begin() + i);
    }
    size_t left()
    {
        return this->size() - i;
    }
};

#endif /* TOOLS_POINTERVECTOR_H_ */
