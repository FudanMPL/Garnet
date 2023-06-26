/*
 * Memory.h
 *
 */

#ifndef GC_MEMORY_H_
#define GC_MEMORY_H_

#include <vector>
#include <sstream>
#include <iostream>
#include <typeinfo>
using namespace std;

#include "Tools/Exceptions.h"
#include "Clear.h"
#include "config.h"

class NoMemory
{
public:
    void resize_min(size_t, const char*) {}
};

namespace GC
{

template <class T>
class Memory : public vector<T>
{
public:
    void resize(size_t size, const char* name = "");
    void resize_min(size_t size, const char* name = "");
    void check_index(Integer index) const;
    T& operator[] (Integer i);
    const T& operator[] (Integer i) const;
    size_t capacity_in_bytes() const { return this->capacity() * sizeof(T); }
};

template <class T>
inline void Memory<T>::check_index(Integer index) const
{
    (void)index;
#ifdef CHECK_SIZE
	size_t i = index.get();
    if (i >= vector<T>::size())
    {
        stringstream ss;
        ss << T::type_string() << " memory overflow: " << i << "/" << vector<T>::size();
        throw Processor_Error(ss.str());
    }
#ifdef DEBUG_MEMORY
    cout << typeid(T).name() << " at " << this << " index " << i << ": "
            << vector<T>::operator[](i) << endl;
#endif
#endif
}

template <class T>
inline T& Memory<T>::operator[] (Integer i)
{
    check_index(i);
    return vector<T>::operator[](i.get());
}

template <class T>
inline const T& Memory<T>::operator[] (Integer i) const
{
    check_index(i);
    return vector<T>::operator[](i.get());
}

template <class T>
inline void Memory<T>::resize(size_t size, const char* name)
{
    (void) name;
#ifdef DEBUG_MEMORY
    if (size > 1000)
        cerr << "Resizing " << T::type_string() << " " << name << " to " << size << endl;
#endif
    vector<T>::resize(size);
}

template <class T>
inline void Memory<T>::resize_min(size_t size, const char* name)
{
    if (this->size() < size)
        resize(size, name);
}

template <class T>
inline ostream& operator<<(ostream& s, const Memory<T>& memory)
{
    for (auto& x : memory)
        x.output(s, false);
    return s;
}

} /* namespace GC */

#endif /* GC_MEMORY_H_ */
