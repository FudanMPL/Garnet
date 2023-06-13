/*
 * FloatInput.h
 *
 */

#ifndef PROCESSOR_FLOATINPUT_H_
#define PROCESSOR_FLOATINPUT_H_

#include "Math/bigint.h"

#include <iostream>

class FloatInput
{
public:
    const static int N_DEST = 4;
    const static int N_PARAM = 1;
    const static char* NAME;

    const static int TYPE = 2;

    long items[N_DEST];

    void read(std::istream& in, const int* params);
};

#endif /* PROCESSOR_FLOATINPUT_H_ */
