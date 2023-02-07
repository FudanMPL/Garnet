/*
 * YaoGarbleInput.h
 *
 */

#ifndef YAO_YAOGARBLEINPUT_H_
#define YAO_YAOGARBLEINPUT_H_

#include "YaoGarbler.h"

class YaoGarbleWire;

class YaoGarbleInput
{
public:
    YaoGarbler& garbler;

    YaoGarbleInput() :
            garbler(YaoGarbler::s())
    {
    }

    void exchange()
    {
        garbler.receiver_input_keys.push_back({});
    }
};

#endif /* YAO_YAOGARBLEINPUT_H_ */
