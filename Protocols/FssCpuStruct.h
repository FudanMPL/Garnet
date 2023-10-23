#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include "Math/bigint.h"


#ifndef PROTOCOLS_FSSCPUSTRUCT_HPP_
#define PROTOCOLS_FSSCPUSTRUCT_HPP_

#define LAMBDA_BYTE 16

class FssEval
{
public:
    octet seed[LAMBDA_BYTE];
    bigint v_hat[2];
    bigint s_hat[2];
    bool t_hat[2];
    bool t[2];
    bool pre_t;
    bigint s[2];
    std::vector<bigint> scw;
    std::vector<bigint> vcw;
    std::vector<bool> tcw[2];
    bigint tmp_v;
    bigint cw;
    bigint convert[2];
};

#endif