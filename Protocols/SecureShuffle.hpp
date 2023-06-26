/*
 * SecureShuffle.hpp
 *
 */

#ifndef PROTOCOLS_SECURESHUFFLE_HPP_
#define PROTOCOLS_SECURESHUFFLE_HPP_

#include "SecureShuffle.h"
#include "Tools/Waksman.h"

#include <math.h>
#include <algorithm>

template<class T>
SecureShuffle<T>::SecureShuffle(SubProcessor<T>& proc) :
        proc(proc), unit_size(0), n_shuffle(0), exact(false)
{
}

template<class T>
SecureShuffle<T>::SecureShuffle(vector<T>& a, size_t n, int unit_size,
        size_t output_base, size_t input_base, SubProcessor<T>& proc) :
        proc(proc), unit_size(unit_size), n_shuffle(0), exact(false)
{
    pre(a, n, input_base);

    for (auto i : proc.protocol.get_relevant_players())
        player_round(i);

    post(a, n, output_base);
}

template<class T>
void SecureShuffle<T>::apply(vector<T>& a, size_t n, int unit_size, size_t output_base,
        size_t input_base, int handle, bool reverse)
{
    this->unit_size = unit_size;

    pre(a, n, input_base);

    auto& shuffle = shuffles.at(handle);
    assert(shuffle.size() == proc.protocol.get_relevant_players().size());

    if (reverse)
        for (auto it = shuffle.end(); it > shuffle.begin(); it--)
        {
            this->config = *(it - 1);
            iter_waksman(reverse);
        }
    else
        for (auto& config : shuffle)
        {
            this->config = config;
            iter_waksman(reverse);
        }

    post(a, n, output_base);
}


template<class T>
void SecureShuffle<T>::inverse_permutation(vector<T> &stack, size_t n, size_t output_base,
                                           size_t input_base) {
    int alice = 0;
    int bob = 1;

    auto &P = proc.P;
    auto &input = proc.input;

    // This method only supports two players
    assert(P.num_players() == 2);
    // The current implementation assumes a semi-honest environment
    assert(!T::malicious);

    // We are dealing directly with permutations, so the unit_size will always be 1.
    this->unit_size = 1;
    // We need to account for sizes which are not a power of 2
    size_t n_pow2 = (1u << int(ceil(log2(n))));

    // Copy over the input registers
    pre(stack, n, input_base);
    // Alice generates stack local permutation and shares the waksman configuration bits secretly to Bob.
    vector<int> perm_alice(n_pow2);
    if (P.my_num() == alice)
        perm_alice = generate_random_permutation(n);
    configure(alice, &perm_alice, n);
    // Apply perm_alice to perm_alice to get perm_bob,
    // stack permutation that we can reveal to Bob without Bob learning anything about perm_alice (since it is masked by perm_a)
    iter_waksman(true);
    // Store perm_bob at stack[output_base]
    post(stack, n, output_base);

    // Reveal permutation perm_bob = perm_a * perm_alice
    // Since this permutation is masked by perm_a, Bob learns nothing about perm
    vector<int> perm_bob(n_pow2);
    typename T::PrivateOutput output(proc);
    for (size_t i = 0; i < n; i++)
        output.prepare_sending(stack[output_base + i], bob);
    output.exchange();
    for (size_t i = 0; i < n; i++) {
        // TODO: Is there a better way to convert a T::clear to int?
        bigint val;
        output.finalize(bob).to(val);
        perm_bob[i] = (int) val.get_si();
    }

    vector<int> perm_bob_inv(n_pow2);
    if (P.my_num() == bob) {
        for (int i = 0; i < (int) n; i++)
            perm_bob_inv[perm_bob[i]] = i;
        // Pad the permutation to n_pow2
        // Required when using waksman networks
        for (int i = (int) n; i < (int) n_pow2; i++)
            perm_bob_inv[i] = i;
    }

    // Alice secret shares perm_a with bob
    // perm_a is stored in the stack at output_base
    input.reset_all(P);
    if (P.my_num() == alice) {
        for (int i = 0; i < (int) n; i++)
            input.add_mine(perm_alice[i]);
    }
    input.exchange();
    for (int i = 0; i < (int) n; i++)
        stack[output_base + i] = input.finalize(alice);

    // The two parties now jointly compute perm_a * perm_bob_inv to obtain perm_inv
    pre(stack, n, output_base);
    configure(bob, &perm_bob_inv, n);
    iter_waksman(true);
    // perm_inv is written back to stack[output_base]
    post(stack, n, output_base);
}

template<class T>
void SecureShuffle<T>::del(int handle)
{
    shuffles.at(handle).clear();
}

template<class T>
void SecureShuffle<T>::pre(vector<T>& a, size_t n, size_t input_base)
{
    n_shuffle = n / unit_size;
    assert(unit_size * n_shuffle == n);
    size_t n_shuffle_pow2 = (1u << int(ceil(log2(n_shuffle))));
    exact = (n_shuffle_pow2 == n_shuffle) or not T::malicious;
    to_shuffle.clear();

    if (exact)
    {
        to_shuffle.resize(n_shuffle_pow2 * unit_size);
        for (size_t i = 0; i < n; i++)
            to_shuffle[i] = a[input_base + i];
    }
    else
    {
        // sorting power of two elements together with indicator bits
        to_shuffle.resize((unit_size + 1) << int(ceil(log2(n_shuffle))));
        for (size_t i = 0; i < n_shuffle; i++)
        {
            for (int j = 0; j < unit_size; j++)
                to_shuffle[i * (unit_size + 1) + j] = a[input_base
                        + i * unit_size + j];
            to_shuffle[i * (unit_size + 1) + unit_size] = T::constant(1,
                    proc.P.my_num(), proc.MC.get_alphai());
        }
        this->unit_size++;
    }
}

template<class T>
void SecureShuffle<T>::post(vector<T>& a, size_t n, size_t output_base)
{
    if (exact)
        for (size_t i = 0; i < n; i++)
            a[output_base + i] = to_shuffle[i];
    else
    {
        auto& MC = proc.MC;
        MC.init_open(proc.P);
        int shuffle_unit_size = this->unit_size;
        int unit_size = shuffle_unit_size - 1;
        for (size_t i = 0; i < to_shuffle.size() / shuffle_unit_size; i++)
            MC.prepare_open(to_shuffle.at((i + 1) * shuffle_unit_size - 1));
        MC.exchange(proc.P);
        size_t i_shuffle = 0;
        for (size_t i = 0; i < n_shuffle; i++)
        {
            auto bit = MC.finalize_open();
            if (bit == 1)
            {
                // only output real elements
                for (int j = 0; j < unit_size; j++)
                    a.at(output_base + i_shuffle * unit_size + j) =
                            to_shuffle.at(i * shuffle_unit_size + j);
                i_shuffle++;
            }
        }
        if (i_shuffle != n_shuffle)
            throw runtime_error("incorrect shuffle");
    }
}

template<class T>
vector<int> SecureShuffle<T>::generate_random_permutation(int n) {
    vector<int> perm;
    int n_pow2 = 1 << int(ceil(log2(n)));
    int shuffle_size = n;
    for (int j = 0; j < n_pow2; j++)
        perm.push_back(j);
    SeededPRNG G;
    for (int i = 0; i < shuffle_size; i++) {
        int j = G.get_uint(shuffle_size - i);
        swap(perm[i], perm[i + j]);
    }

    return perm;
}

template<class T>
void SecureShuffle<T>::player_round(int config_player) {
    vector<int> random_perm(n_shuffle);
    if (proc.P.my_num() == config_player)
        random_perm = generate_random_permutation(n_shuffle);
    configure(config_player, &random_perm, n_shuffle);
    iter_waksman();
}

template<class T>
int SecureShuffle<T>::generate(int n_shuffle)
{
    int res = shuffles.size();
    shuffles.push_back({});
    auto& shuffle = shuffles.back();

    for (auto i: proc.protocol.get_relevant_players()) {
        vector<int> perm;
        if (proc.P.my_num() == i)
            perm = generate_random_permutation(n_shuffle);
        configure(i, &perm, n_shuffle);

        shuffle.push_back(config);
    }

    return res;
}

template<class T>
void SecureShuffle<T>::configure(int config_player, vector<int> *perm, int n) {
    auto &P = proc.P;
    auto &input = proc.input;
    input.reset_all(P);
    int n_pow2 = 1 << int(ceil(log2(n)));
    Waksman waksman(n_pow2);

    // The player specified by config_player configures the shared waksman network
    // using its personal permutation
    if (P.my_num() == config_player) {
        auto config_bits = waksman.configure(*perm);
        for (size_t i = 0; i < config_bits.size(); i++) {
            auto &x = config_bits[i];
            for (size_t j = 0; j < x.size(); j++)
                if (waksman.matters(i, j) and not waksman.is_double(i, j))
                    input.add_mine(int(x[j]));
                else if (waksman.is_double(i, j))
                    assert(x[j] == x[j - 1]);
                else
                    assert(x[j] == 0);
        }
        // The other player waits for its share of the configured waksman network
    } else
        for (size_t i = 0; i < waksman.n_bits(); i++)
            input.add_other(config_player);

    input.exchange();
    config.clear();
    typename T::Protocol checker(P);
    checker.init(proc.DataF, proc.MC);
    checker.init_dotprod();
    auto one = T::constant(1, P.my_num(), proc.MC.get_alphai());
    for (size_t i = 0; i < waksman.n_rounds(); i++)
    {
        config.push_back({});
        for (int j = 0; j < n_pow2; j++)
        {
            if (waksman.matters(i, j) and not waksman.is_double(i, j))
            {
                config.back().push_back(input.finalize(config_player));
                if (T::malicious)
                    checker.prepare_dotprod(config.back().back(),
                            one - config.back().back());
            }
            else if (waksman.is_double(i, j))
                config.back().push_back(config.back().back());
            else
                config.back().push_back({});
        }
    }

    if (T::malicious)
    {
        checker.next_dotprod();
        checker.exchange();
        assert(
                typename T::clear(
                        proc.MC.open(checker.finalize_dotprod(waksman.n_bits()),
                                P)) == 0);
        checker.check();
    }
}

template<class T>
void SecureShuffle<T>::waksman(vector<T>& a, int depth, int start)
{
    int n = a.size();

    if (n == 2)
    {
        cond_swap(a[0], a[1], config.at(depth).at(start));
        return;
    }

    vector<T> a0(n / 2), a1(n / 2);
    for (int i = 0; i < n / 2; i++)
    {
        a0.at(i) = a.at(2 * i);
        a1.at(i) = a.at(2 * i + 1);

        cond_swap(a0[i], a1[i], config.at(depth).at(i + start + n / 2));
    }

    waksman(a0, depth + 1, start);
    waksman(a1, depth + 1, start + n / 2);

    for (int i = 0; i < n / 2; i++)
    {
        a.at(2 * i) = a0.at(i);
        a.at(2 * i + 1) = a1.at(i);
        cond_swap(a[2 * i], a[2 * i + 1], config.at(depth).at(i + start));
    }
}

template<class T>
void SecureShuffle<T>::cond_swap(T& x, T& y, const T& b)
{
    auto diff = proc.protocol.mul(x - y, b);
    x -= diff;
    y += diff;
}

template<class T>
void SecureShuffle<T>::iter_waksman(bool reverse)
{
    int n = to_shuffle.size() / unit_size;

    for (int depth = 0; depth < log2(n); depth++)
        waksman_round(depth, true, reverse);

    for (int depth = log2(n) - 2; depth >= 0; depth--)
        waksman_round(depth, false, reverse);
}

template<class T>
void SecureShuffle<T>::waksman_round(int depth, bool inwards, bool reverse)
{
    int n = to_shuffle.size() / unit_size;
    assert((int) config.at(depth).size() == n);
    int nblocks = 1 << depth;
    int size = n / (2 * nblocks);
    bool outwards = !inwards;
    proc.protocol.init_mul();
    vector<array<int, 5>> indices;
    indices.reserve(n / 2);
    Waksman waksman(n);
    for (int k = 0; k < n / 2; k++)
    {
        int j = k % size;
        int i = k / size;
        int base = 2 * i * size;
        int in1 = base + j + j * inwards;
        int in2 = in1 + inwards + size * outwards;
        int out1 = base + j + j * outwards;
        int out2 = out1 + outwards + size * inwards;
        int i_bit = base + j + size * (outwards ^ reverse);
        bool run = waksman.matters(depth, i_bit);
        if (run)
        {
            for (int l = 0; l < unit_size; l++)
                proc.protocol.prepare_mul(config.at(depth).at(i_bit),
                        to_shuffle.at(in1 * unit_size + l)
                                - to_shuffle.at(in2 * unit_size + l));
        }
        indices.push_back({{in1, in2, out1, out2, run}});
    }
    proc.protocol.exchange();
    tmp.resize(to_shuffle.size());
    for (int k = 0; k < n / 2; k++)
    {
        auto idx = indices.at(k);
        for (int l = 0; l < unit_size; l++)
        {
            T diff;
            if (idx[4])
                diff = proc.protocol.finalize_mul();
            tmp.at(idx[2] * unit_size + l) = to_shuffle.at(
                    idx[0] * unit_size + l) - diff;
            tmp.at(idx[3] * unit_size + l) = to_shuffle.at(
                    idx[1] * unit_size + l) + diff;
        }
    }
    swap(tmp, to_shuffle);
}

#endif /* PROTOCOLS_SECURESHUFFLE_HPP_ */
