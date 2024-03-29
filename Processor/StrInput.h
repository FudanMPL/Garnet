/*
 * StrInput.h
 *
 */

#ifndef PROCESSOR_STRINPUT_H_
#define PROCESSOR_STRINPUT_H_

#include <iostream>

#include "Math/bigint.h"
#include "Math/Integer.h"

template<class T>
class StrInput_
{
public:
    const static int N_DEST = 1;
    const static int N_PARAM = 0;
    const static char* NAME;

    const static int TYPE = 3;

    T items[N_DEST];

    void read(std::istream& in, const int* params);
};

template<class T>
const char* StrInput_<T>::NAME = "real number";

#ifdef LOW_PREC_INP
typedef StrInput_<Integer> StrInput;
#else
typedef StrInput_<bigint> StrInput;
#endif

#endif /* PROCESSOR_STRINPUT_H_ */
