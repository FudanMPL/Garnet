/*
 * Abfllnoww.h
 *
 */

#ifndef GC_POSTSACRIBIN_H_
#define GC_POSTSACRIBIN_H_

#include "PostSacriSecret.h"
#include "Protocols/Replicated.h"
#include "ShiftableTripleBuffer.h"

namespace GC
{

class PostSacriBin : public ReplicatedBase,
        public ProtocolBase<PostSacriSecret>,
        ShiftableTripleBuffer<PostSacriSecret>
{
    typedef PostSacriSecret T;

    Replicated<T> honest;

    vector<array<T, 2>> inputs;
    vector<pair<T, int>> outputs;

    // as in Araki et al. (S&P'17)
    vector<FixedVec<T, 3>> d1;
    vector<array<T, 3>> d2;

    array<T, 3> get_d1_triple(GlobalPRNG& G, int n_bits);
    array<T, 3> get_d2_triple(int n_bits);
    void get(Dtype type, T* res);

    const size_t N = 1 << 20;

public:
    PostSacriBin(Player& P);
    ~PostSacriBin();

    void init_mul();
    void prepare_mul(const T& x, const T& y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);

    void check();
};

} /* namespace GC */

#endif /* GC_POSTSACRIBIN_H_ */
