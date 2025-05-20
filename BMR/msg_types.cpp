/*
 * msg_types.cpp
 *
 */

#include "msg_types.h"

#define X(NAME) #NAME,

const char* message_type_names[] = {
		MESSAGE_TYPES
};

#undef X
