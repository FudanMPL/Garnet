/*
 * common.h
 *
 */

#ifndef NETWORK_INC_COMMON_H_
#define NETWORK_INC_COMMON_H_

#include <string>
#include <stdexcept>

//#include "utils.h"

/*
 * To change the buffer sizes in the kernel
# echo 'net.core.wmem_max=12582912' >> /etc/sysctl.conf
# echo 'net.core.rmem_max=12582912' >> /etc/sysctl.conf
*/
const int NETWORK_BUFFER_SIZE = 20000000;

typedef struct {
	std::string ip;
	int port;
} endpoint_t;

#endif /* NETWORK_INC_COMMON_H_ */
