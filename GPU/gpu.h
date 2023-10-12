#include "Math/bigint.h"
#include <iostream>


void add(uint8_t *a, uint8_t *b, uint8_t *res, int length);

void restricted_multiply(int value, uint8_t * a, uint8_t *res, int length);

void test_add(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void test_sub(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void test_restricted_multiply(int value, uint8_t * a, uint8_t * res, int numbytes);

void test_xor(uint8_t * a, uint8_t * b, uint8_t * res, int numbytes);

void fss_generate(uint8_t * r, uint8_t * seed0, uint8_t * seed1, uint8_t * generated_value_cpu, int numbytes, int parallel);

void fss_evaluate(int party, uint8_t * x_reveal, uint8_t * seed, uint8_t * gen_val, uint8_t * result, int numbytes, int parallel);