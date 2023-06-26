/*
 * callgrind.h
 *
 */

#ifndef TOOLS_CALLGRIND_H_
#define TOOLS_CALLGRIND_H_

#ifdef USE_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#define CALLGRIND_DUMP_STATS
#endif

#endif /* TOOLS_CALLGRIND_H_ */
