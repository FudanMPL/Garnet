/*
 * CheckVector.h
 *
 */

#ifndef TOOLS_CHECKVECTOR_H_
#define TOOLS_CHECKVECTOR_H_

#include <vector>
using namespace std;

template <class T>
class CheckVector : public vector<T>
{
public:
    CheckVector() : vector<T>() {}
    CheckVector(size_t size) : vector<T>(size) {}
    CheckVector(size_t size, const T& def) : vector<T>(size, def) {}
#ifdef CHECK_SIZE
    T& operator[](size_t i) { return this->at(i); }
    const T& operator[](size_t i) const { return this->at(i); }
#else
    T& at(size_t i) { return (*this)[i]; }
    const T& at(size_t i) const { return (*this)[i]; }
#endif
};

#endif /* TOOLS_CHECKVECTOR_H_ */
