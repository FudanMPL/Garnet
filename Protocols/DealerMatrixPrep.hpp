/*
 * DealerMatrixPrep.hpp
 *
 */

#include "DealerMatrixPrep.h"

template<class T>
DealerMatrixPrep<T>::DealerMatrixPrep(int n_rows, int n_inner, int n_cols,
        typename T::LivePrep& prep, DataPositions& usage) :
        super(usage), n_rows(n_rows), n_inner(n_inner), n_cols(n_cols),
        prep(&prep)
{
}

template<class T>
void append_shares(vector<octetStream>& os,
        ValueMatrix<typename T::clear>& M, PRNG& G)
{
    size_t n = os.size();
    for (auto& value : M.entries)
    {
        T sum;
        for (size_t i = 0; i < n - 2; i++)
        {
            auto share = G.get<T>();
            sum += share;
            share.pack(os[i]);
        }
        (value - sum).pack(os[n - 2]);
    }
}

template<class T>
ShareMatrix<T> receive_shares(octetStream& o, int n, int m)
{
    ShareMatrix<T> res(n, m);
    for (size_t i = 0; i < res.entries.size(); i++)
        res.entries.v.push_back(o.get<T>());
    return res;
}

template<class T>
void DealerMatrixPrep<T>::buffer_triples()
{
    assert(this->prep);
    assert(this->prep->proc);
    auto& P = this->prep->proc->P;
    vector<bool> senders(P.num_players());
    senders.back() = true;
    octetStreams os(P), to_receive(P);
    int batch_size = 100;
    if (not T::real_shares(P))
    {
        SeededPRNG G;
        ValueMatrix<typename T::clear> A(n_rows, n_inner), B(n_inner, n_cols),
                C(n_rows, n_cols);
        for (int i = 0; i < P.num_players() - 1; i++)
            os[i].reserve(
                    batch_size * T::size()
                            * (A.entries.size() + B.entries.size()
                                    + C.entries.size()));
        for (int i = 0; i < batch_size; i++)
        {
            A.randomize(G);
            B.randomize(G);
            C = A * B;
            append_shares<T>(os, A, G);
            append_shares<T>(os, B, G);
            append_shares<T>(os, C, G);
            this->triples.push_back({{{n_rows, n_inner}, {n_inner, n_cols},
                {n_rows, n_cols}}});
        }
        P.send_receive_all(senders, os, to_receive);
    }
    else
    {
        P.send_receive_all(senders, os, to_receive);
        for (int i = 0; i < batch_size; i++)
        {
            auto& o = to_receive.back();
            this->triples.push_back({{receive_shares<T>(o, n_rows, n_inner),
                receive_shares<T>(o, n_inner, n_cols),
                receive_shares<T>(o, n_rows, n_cols)}});
        }
    }
}
