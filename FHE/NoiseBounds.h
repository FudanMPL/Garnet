/*
 * NoiseBound.h
 *
 */

#ifndef FHE_NOISEBOUNDS_H_
#define FHE_NOISEBOUNDS_H_

#include "Math/bigint.h"

int phi_N(int N);
class FHE_Params;

class SemiHomomorphicNoiseBounds
{
protected:
    static const int FHE_epsilon = 55;

    const bigint p;
    const int phi_m;
    const int n;
    const int sec;
    int slack;
    mpf_class sigma;

    bigint B_clean;
    bigint B_scale;
    bigint drown;

    mpf_class c1, c2;
    mpf_class V_s;

    void produce_epsilon_constants();

public:
    SemiHomomorphicNoiseBounds(const bigint& p, int phi_m, int n, int sec,
            int slack, bool extra_h, const FHE_Params& params);
    // with scaling
    bigint min_p0(const bigint& p1);
    // without scaling
    bigint min_p0();
    bigint min_p0(bool scale, const bigint& p1) { return scale ? min_p0(p1) : min_p0(); }
    static double min_phi_m(int log_q, double sigma);
    static double min_phi_m(int log_q, const FHE_Params& params);

    bigint get_B_clean() { return B_clean; }
};

// as per ePrint 2012:642 for slack = 0
class NoiseBounds : public SemiHomomorphicNoiseBounds
{
    bigint B_KS;

public:
    NoiseBounds(const bigint& p, int phi_m, int n, int sec, int slack,
            const FHE_Params& params);
    bigint U1(const bigint& p0, const bigint& p1);
    bigint U2(const bigint& p0, const bigint& p1);
    bigint min_p0(const bigint& p0, const bigint& p1);
    bigint min_p0(const bigint& p1);
    bigint min_p1();
    bigint opt_p1();
    bigint opt_p0() { return min_p0(opt_p1()); }
    double optimize(int& lg2p0, int& lg2p1);
};

#endif /* FHE_NOISEBOUNDS_H_ */
