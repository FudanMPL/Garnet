/*
 * Rep3Shuffler.cpp
 *
 */

#ifndef PROTOCOLS_REP3SHUFFLER_HPP_
#define PROTOCOLS_REP3SHUFFLER_HPP_

#include "Rep3Shuffler.h"

template<class T>
Rep3Shuffler<T>::Rep3Shuffler(vector<T>& a, size_t n, int unit_size,
        size_t output_base, size_t input_base, SubProcessor<T>& proc) :
        proc(proc)
{
    apply(a, n, unit_size, output_base, input_base, generate(n / unit_size),
            false);
    shuffles.pop_back();
}

template<class T>
Rep3Shuffler<T>::Rep3Shuffler(SubProcessor<T>& proc) :
        proc(proc)
{
}

template<class T>
int Rep3Shuffler<T>::generate(int n_shuffle)
{
    shuffles.push_back({});
    auto& shuffle = shuffles.back();
    for (int i = 0; i < 2; i++)
    {
        auto& perm = shuffle[i];
        for (int j = 0; j < n_shuffle; j++)
            perm.push_back(j);
        for (int j = 0; j < n_shuffle; j++)
        {
            int k = proc.protocol.shared_prngs[i].get_uint(n_shuffle - j);
            swap(perm[k], perm[k + j]);
        }
    }
    return shuffles.size() - 1;
}

template<class T>
void Rep3Shuffler<T>::apply(vector<T>& a, size_t n, int unit_size,
        size_t output_base, size_t input_base, int handle, bool reverse)
{
    assert(proc.P.num_players() == 3);
    assert(not T::malicious);
    assert(not T::dishonest_majority);
    assert(n % unit_size == 0);

    auto& shuffle = shuffles.at(handle);
    vector<T> to_shuffle;
    for (size_t i = 0; i < n; i++)
        to_shuffle.push_back(a[input_base + i]);

    typename T::Input input(proc);

    vector<typename T::clear> to_share(n);

    for (int ii = 0; ii < 3; ii++)
    {
        int i;
        if (reverse)
            i = 2 - ii;
        else
            i = ii;

        if (proc.P.get_player(i) == 0)
        {
            for (size_t j = 0; j < n / unit_size; j++)
                for (int k = 0; k < unit_size; k++)
                    if (reverse)
                        to_share.at(j * unit_size + k) = to_shuffle.at(
                                shuffle[0].at(j) * unit_size + k).sum();
                    else
                        to_share.at(shuffle[0].at(j) * unit_size + k) =
                                to_shuffle.at(j * unit_size + k).sum();
        }
        else if (proc.P.get_player(i) == 1)
        {
            for (size_t j = 0; j < n / unit_size; j++)
                for (int k = 0; k < unit_size; k++)
                    if (reverse)
                        to_share[j * unit_size + k] = to_shuffle[shuffle[1][j]
                                * unit_size + k][0];
                    else
                        to_share[shuffle[1][j] * unit_size + k] = to_shuffle[j
                                * unit_size + k][0];
        }

        input.reset_all(proc.P);

        if (proc.P.get_player(i) < 2)
            for (auto& x : to_share)
                input.add_mine(x);

        for (int k = 0; k < 2; k++)
            input.add_other((-i + 3 + k) % 3);

        input.exchange();
        to_shuffle.clear();

        for (size_t j = 0; j < n; j++)
        {
            T x = input.finalize((-i + 3) % 3) + input.finalize((-i + 4) % 3);
            to_shuffle.push_back(x);
        }
    }

    for (size_t i = 0; i < n; i++)
        a[output_base + i] = to_shuffle[i];
}

template<class T>
void Rep3Shuffler<T>::del(int handle)
{
    for (int i = 0; i < 2; i++)
        shuffles.at(handle)[i].clear();
}

template<class T>
void Rep3Shuffler<T>::inverse_permutation(vector<T>&, size_t, size_t, size_t)
{
    throw runtime_error("inverse permutation not implemented");
}

#endif
