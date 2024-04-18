/*
 * IntInput.cpp
 *
 */

#ifndef PROCESSOR_INTINPUT_HPP_
#define PROCESSOR_INTINPUT_HPP_

#include "IntInput.h"

template<class T>
const char* IntInput<T>::NAME = "integer";

template<class T>
void IntInput<T>::read(std::istream& in, const int*)
{
    in >> items[0];
}

#endif
