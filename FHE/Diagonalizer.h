/*
 * Diagonalizer.h
 *
 */

#ifndef FHE_DIAGONALIZER_H_
#define FHE_DIAGONALIZER_H_

#include "Math/gfpvar.h"
#include "Ciphertext.h"
#include "Protocols/ShareMatrix.h"

class Diagonalizer
{
    const FFT_Data& FTD;

    size_t n_rows, n_cols;

public:
    typedef AddableVector<ValueMatrix<gfpvar>> MatrixVector;

    vector<Ciphertext> ciphertexts;

    Diagonalizer(const MatrixVector& matrices,
            const FFT_Data& FTD, const FHE_PK& pk);

    Plaintext_<FFT_Data> get_plaintext(const MatrixVector& matrices,
            int left_col, int right_col);

    MatrixVector decrypt(const vector<Ciphertext>&, int n_matrices, FHE_SK& sk);

    MatrixVector dediag(const vector<Plaintext_<FFT_Data>>& plaintexts,
            int n_matrices);
};

#endif /* FHE_DIAGONALIZER_H_ */
