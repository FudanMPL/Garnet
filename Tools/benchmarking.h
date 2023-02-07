/*
 * benchmarking.h
 *
 */

#ifndef TOOLS_BENCHMARKING_H_
#define TOOLS_BENCHMARKING_H_

#include <stdexcept>
#include <string>
#include <iostream>
using namespace std;

// call before insecure benchmarking functionality
void insecure(string message, bool warning = true);

void insecure_fake(bool warning = true);

#endif /* TOOLS_BENCHMARKING_H_ */
