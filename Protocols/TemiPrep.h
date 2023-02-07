/*
 * TemiPrep.h
 *
 */

#ifndef PROTOCOLS_TEMIPREP_H_
#define PROTOCOLS_TEMIPREP_H_

#include "ReplicatedPrep.h"
#include "FHEOffline/TemiSetup.h"

template<class T> class HemiMatrixPrep;

template<class T>
class TemiMultiplier
{
    typedef typename T::clear::FD FD;

    vector<Ciphertext> multiplicands;

    Player& P;

public:
    TemiMultiplier(Player& P);

    vector<Ciphertext>& get_multiplicands(
            vector<vector<Ciphertext>>& ciphertexts, const FHE_PK& pk);
    void add(Plaintext_<FD>& res, const Ciphertext& C, OT_ROLE role = BOTH,
            int n_summands = 1);

    int get_offset()
    {
        return 0;
    }
};

/**
 * Semi-honest triple generation with semi-homomorphic encryption
 */
template<class T>
class TemiPrep : public SemiHonestRingPrep<T>
{
    friend class HemiMatrixPrep<T>;

    typedef typename T::clear::FD FD;

    static Lock lock;
    static TemiSetup<FD>* setup;

    vector<TemiMultiplier<T>*> multipliers;

public:
    static void basic_setup(Player& P);
    static void teardown();

    static const FD& get_FTD();
    static const FHE_PK& get_pk();
    static const TemiSetup<FD>& get_setup();

    TemiPrep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage),
            BitPrep<T>(proc, usage), RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage)
    {
    }

    ~TemiPrep();

    void buffer_triples();

    vector<TemiMultiplier<T>*>& get_multipliers();
};

#endif /* PROTOCOLS_TEMIPREP_H_ */
