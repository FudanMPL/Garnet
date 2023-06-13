/*
 * utils.cpp
 *
 */



#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
//#include <intrin.h>
#include <stdlib.h>

#include "proto_utils.h"

void fill_message_type(void* buffer, MSG_TYPE type)
{
	memcpy(buffer, &type, sizeof(MSG_TYPE));
}

void fill_message_type(SendBuffer& buffer, MSG_TYPE type)
{
	buffer.serialize(type);
}
