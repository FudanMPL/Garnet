/*
 * FixedVector.h
 *
 */

#ifndef TOOLS_FIXEDVECTOR_H_
#define TOOLS_FIXEDVECTOR_H_

#include <array>
#include <vector>
#include <assert.h>
using namespace std;

template<class T, size_t L>
class FixedVector : array<T, L>
{
    size_t used_size;

public:
    static const size_t MAX_SIZE = L;

    FixedVector()
    {
        used_size = 0;
    }

    FixedVector(const vector<T>& other)
    {
        reserve(other.size());
        std::copy(other.begin(), other.end(), this->begin());
        used_size = other.size();
    }

    void reserve(size_t size)
    {
        if (size > L)
            throw overflow("FixedVector", size, L);
    }

    void push_back(const T& x)
    {
        assert(used_size < L);
        push_back_no_check(x);
    }

    void push_back_no_check(const T& x)
    {
        (*this)[used_size] = x;
        used_size++;
    }

    T pop()
    {
        assert(used_size > 0);
        return (*this)[--used_size];
    }

    bool empty()
    {
        return used_size == 0;
    }

    bool full()
    {
        return used_size == L;
    }

    size_t size() const
    {
        return used_size;
    }

    void resize(size_t size, const T& def = {})
    {
        assert(size <= L);
        for (size_t i = used_size; i < size; i++)
            (*this)[i] = def;
        used_size = size;
    }

    void clear()
    {
        used_size = 0;
    }

    typename array<T, L>::iterator begin()
    {
        return array<T, L>::begin();
    }

    typename array<T, L>::iterator end()
    {
        return begin() + used_size;
    }

    typename array<T, L>::const_iterator begin() const
    {
        return array<T, L>::begin();
    }

    typename array<T, L>::const_iterator end() const
    {
        return begin() + used_size;
    }

    T& operator[](size_t i)
    {
        return array<T, L>::operator[](i);
    }

    const T& operator[](size_t i) const
    {
        return array<T, L>::operator[](i);
    }

    T& at(size_t i)
    {
        return array<T, L>::at(i);
    }

    const T& at(size_t i) const
    {
        return array<T, L>::at(i);
    }
};

#endif /* TOOLS_FIXEDVECTOR_H_ */
