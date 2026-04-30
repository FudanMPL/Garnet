/*
 * intrinsics.h
 *
 */

#ifndef TOOLS_INTRINSICS_H_
#define TOOLS_INTRINSICS_H_

#ifdef __x86_64__
#include <immintrin.h>
#include <x86intrin.h>
#else
#ifdef __aarch64__
#define SIMDE_X86_AVX_ENABLE_NATIVE_ALIASES
#define SIMDE_X86_AVX2_ENABLE_NATIVE_ALIASES
#include "simde/simde/x86/avx2.h"
#include "simde/simde/x86/clmul.h"
#define SSE2NEON_SUPPRESS_WARNINGS
#include "sse2neon/sse2neon.h"
#endif
#endif

#endif /* TOOLS_INTRINSICS_H_ */
