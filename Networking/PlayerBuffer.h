/*
 * PlayerBuffer.h
 *
 */

#ifndef NETWORKING_PLAYERBUFFER_H_
#define NETWORKING_PLAYERBUFFER_H_

#include "Tools/int.h"

class PlayerBuffer
{
public:
    octet* data;
    size_t size;

    PlayerBuffer(octet* data, size_t size) :
            data(data), size(size)
    {
    }
};

#endif /* NETWORKING_PLAYERBUFFER_H_ */
