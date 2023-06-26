/*
 * config.h
 *
 */

#ifndef YAO_CONFIG_H_
#define YAO_CONFIG_H_

//#define DEBUG

//#define CHECK_BUFFER

class YaoFullGate;
class YaoHalfGate;

#ifndef FULL_GATES
typedef YaoHalfGate YaoGate;
#else
typedef YaoFullGate YaoGate;
#endif

#endif /* YAO_CONFIG_H_ */
