/*
 * utils.h
 *
 */

#ifndef PROTO_UTILS_H_
#define PROTO_UTILS_H_

#include "msg_types.h"
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
using namespace std;

#include "Tools/avx_memcpy.h"
#include "Tools/FlexBuffer.h"

#define LOOPBACK_STR "LOOPBACK"

class SendBuffer;

void fill_message_type(void* buffer, MSG_TYPE type);
void fill_message_type(SendBuffer& buffer, MSG_TYPE type);

void phex (const void *addr, int len);


inline void phex(const FlexBuffer& buffer) { phex(buffer.data(), buffer.size()); }

#endif /* NETWORK_TEST_UTILS_H_ */
