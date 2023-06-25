/*
 * ShuffleSacrifice.hpp
 *
 */

#ifndef PROTOCOLS_SHUFFLESACRIFICE_HPP_
#define PROTOCOLS_SHUFFLESACRIFICE_HPP_

#include "ShuffleSacrifice.h"
#include "Tools/PointerVector.h"
#include "GC/BitAdder.h"

#include "LimitedPrep.hpp"

inline
ShuffleSacrifice::ShuffleSacrifice() :
        ShuffleSacrifice(OnlineOptions::singleton.bucket_size, 3)
{
}

inline
ShuffleSacrifice::ShuffleSacrifice(int B, int C) :
        B(B), C(C)
{
    if (OnlineOptions::singleton.security_parameter > 40)
        throw runtime_error("shuffle sacrifice not implemented for more than "
                "40-bit security");
}

template<class U>
void ShuffleSacrifice::shuffle(vector<U>& check_triples, Player& P)
{
    int buffer_size = check_triples.size();

    // shuffle
    GlobalPRNG G(P);
    for (int i = 0; i < buffer_size; i++)
    {
        int remaining = buffer_size - i;
        int pos = G.get_uint(remaining);
        swap(check_triples[i], check_triples[i + pos]);
    }
}

template<class T>
void TripleShuffleSacrifice<T>::triple_combine(vector<array<T, 3> >& triples,
        vector<array<T, 3> >& to_combine, Player& P,
        typename T::MAC_Check& MC)
{
    int buffer_size = to_combine.size();
    int N = buffer_size / B;
    assert(minimum_n_outputs() <= N);

    shuffle(to_combine, P);

    vector<typename T::open_type> opened;
    vector<T> masked;
    masked.reserve(N);
    for (int i = 0; i < N; i++)
    {
        T& b = to_combine[i][1];
        for (int j = 1; j < B; j++)
        {
            T& g = to_combine[i + N * j][1];
            masked.push_back(b - g);
        }
    }
    MC.POpen(opened, masked, P);
    auto it = opened.begin();
    for (int i = 0; i < N; i++)
    {
        T& a = to_combine[i][0];
        T& c = to_combine[i][2];
        for (int j = 1; j < B; j++)
        {
            T& f = to_combine[i + N * j][0];
            T& h = to_combine[i + N * j][2];
            auto& rho = *(it++);
            a += f;
            c += h + f * rho;
        }
    }
    to_combine.resize(N);
    triples = to_combine;
}

template<class T>
void DabitShuffleSacrifice<T>::dabit_sacrifice(vector<dabit<T> >& output,
        vector<dabit<T> >& to_check, SubProcessor<T>& proc,
        ThreadQueues* queues)
{
#ifdef VERBOSE_DABIT
    cerr << "Sacrificing daBits" << endl;
#endif

    auto& P = proc.P;
    auto& MC = proc.MC;

    int buffer_size = to_check.size();
    int N = (buffer_size - C) / B;

    shuffle(to_check, P);

    // opening C
    vector<T> shares;
    vector<typename T::bit_type::part_type> bit_shares;
    for (int i = 0; i < C; i++)
    {
        shares.push_back(to_check.back().first);
        bit_shares.push_back(to_check.back().second);
        to_check.pop_back();
    }
    vector<typename T::open_type> opened;
    MC.POpen(opened, shares, P);
    vector<typename T::bit_type::part_type::open_type> bits;
    auto& MCB = *T::bit_type::part_type::new_mc(
            GC::ShareThread<typename T::bit_type>::s().MC->get_alphai());
    MCB.POpen(bits, bit_shares, P);
    for (int i = 0; i < C; i++)
        if (typename T::clear(opened[i]) != bits[i].get())
            throw Offline_Check_Error("dabit shuffle opening");

    // sacrifice buckets
    typename T::Protocol protocol(P);
    vector<pair<T, T>> multiplicands;
    for (int i = 0; i < N; i++)
    {
        auto& a = to_check[i].first;
        for (int j = 1; j < B; j++)
        {
            auto& f = to_check[i + N * j].first;
            multiplicands.push_back({a, f});
        }
    }

    PointerVector<T> products;
    products.resize(multiplicands.size());
    if (queues)
    {
        ThreadJob job(&products, &multiplicands);
        int start = queues->distribute(job, products.size());
        protocol.multiply(products, multiplicands,
                start, products.size(), proc);
        if (start)
            queues->wrap_up(job);
    }
    else
        protocol.multiply(products, multiplicands, 0, products.size(), proc);

    shares.clear();
    bit_shares.clear();
    shares.reserve((B - 1) * N);
    bit_shares.reserve((B - 1) * N);
    for (int i = 0; i < N; i++)
    {
        auto& a = to_check[i].first;
        auto& b = to_check[i].second;
        for (int j = 1; j < B; j++)
        {
            auto& f = to_check[i + N * j].first;
            auto& g = to_check[i + N * j].second;
            shares.push_back(a + f - products.next() * 2);
            bit_shares.push_back(b + g);
        }
    }
    MC.POpen(opened, shares, P);
    MCB.POpen(bits, bit_shares, P);
    for (int i = 0; i < (B - 1) * N; i++)
        if (typename T::clear(opened[i]) != bits[i].get())
            throw Offline_Check_Error("dabit shuffle opening");
    MCB.Check(P);

    to_check.resize(N);
    output.insert(output.end(), to_check.begin(), to_check.end());
    delete &MCB;
}

template<class T>
void EdabitShuffleSacrifice<T>::edabit_sacrifice(vector<edabit<T> >& output,
        vector<T>& wholes, vector<vector<typename T::bit_type::part_type>>& parts,
        size_t n_bits, SubProcessor<T>& proc, bool strict, int player,
        ThreadQueues* queues)
{
#ifdef VERBOSE_EDA
    cerr << "Sacrificing edaBits of length " << n_bits << endl;
    Timer timer;
    timer.start();
#endif

    auto& P = proc.P;
    auto& MC = proc.MC;

    typedef typename T::bit_type::part_type BT;
    typedef typename BT::small_type ST;
    vector<edabit<T>> to_check;
    size_t dl = T::bit_type::part_type::default_length;
    assert(wholes.size() >= size_t(minimum_n_inputs(minimum_n_outputs())));
    assert(parts.size() >= n_bits);
    for (auto& x: parts)
        assert(x.size() >= size_t(DIV_CEIL(wholes.size(), dl)));

    RunningTimer init_timer;
    to_check.reserve(wholes.size());
    size_t n_bits_input = parts.size();
    assert(n_bits_input <= edabit<T>::second_type::MAX_SIZE);
    for (int i1 = 0; i1 < DIV_CEIL(wholes.size(), dl); i1++)
    {
        int n = min(dl, wholes.size() - i1 * dl);
        for (int i2 = 0; i2 < n; i2++)
            to_check.push_back({wholes[i1 * dl + i2], {}});
        for (size_t k = 0; k < n_bits_input; k++)
        {
            auto& bits = parts[k][i1];
            for (int i2 = 0; i2 < n; i2++)
                to_check[i1 * dl + i2].second.push_back_no_check(
                        ST(bits >> i2).get_bit(0));
        }
    }
    wholes.clear();
    wholes.shrink_to_fit();
    parts.clear();
    parts.shrink_to_fit();

#ifdef VERBOSE_EDA
    cerr << "Initialization took " << init_timer.elapsed() << " seconds" << endl;
#endif

    int buffer_size = to_check.size();
    int N = (buffer_size - C) / B;

    // needs to happen before shuffling for security
    LimitedPrep<BT> personal_prep;
    RunningTimer personal_timer;
    if (player >= 0)
    {
        auto &party = GC::ShareThread<typename T::bit_type>::s();
        SubProcessor<BT> bit_proc(party.MC->get_part_MC(),
                *proc.personal_bit_preps.at(player), P);
        int n_triples = DIV_CEIL((B - 1) * N * n_bits, dl);
        proc.personal_bit_preps.at(player)->buffer_personal_triples(n_triples,
                queues);
        for (int i = 0; i < n_triples; i++)
            personal_prep.push_triple(
                    proc.personal_bit_preps.at(player)->get_triple(dl));
        proc.personal_bit_preps.at(player)->shrink_to_fit();
    }
#ifdef VERBOSE_EDA
    cerr << "Personal preprocessing took " << personal_timer.elapsed() << " seconds" << endl;
#endif

    RunningTimer shuffle_timer;
    shuffle(to_check, P);
#ifdef VERBOSE_EDA
    cerr << "Shuffling took " << shuffle_timer.elapsed() << " seconds" << endl;
#endif

    // opening C
    vector<T> shares;
    vector<typename BT::small_type> bit_shares;
    for (int i = 0; i < C; i++)
    {
        assert(to_check.back().second.size() * i == bit_shares.size());
        shares.push_back(to_check.back().first);
        bit_shares.insert(bit_shares.end(), to_check.back().second.begin(),
                to_check.back().second.end());
        to_check.pop_back();
    }
    vector<typename T::open_type> opened;
    MC.POpen(opened, shares, P);
    vector<typename BT::small_type::open_type> bits;
    auto& MCB = *BT::small_type::new_mc(
            GC::ShareThread<typename T::bit_type>::s().MC->get_alphai());
    MCB.POpen(bits, bit_shares, P);
    size_t n_bits_to_open = bits.size() / C;
    for (int i = 0; i < C; i++)
    {
        typename T::clear sum, single = opened[i];
        for (size_t j = 0; j < n_bits_to_open; j++)
            sum += typename T::clear(bits.at(i * n_bits_to_open + j).get())
                    << j;
        if (single != sum)
        {
            cout << single << " != " << sum << endl;
            cout << "bits: ";
            for (size_t j = 0; j < n_bits_to_open; j++)
                cout << bits.at(i * n_bits_to_open + j);
            cout << endl;
            throw Offline_Check_Error("edabit shuffle opening");
        }
    }

    RunningTimer bucket_timer;
    if (queues)
    {
        int n_available = queues->find_available();
        int n_per_thread = queues->get_n_per_thread(N, dl);
        vector<vector<array<BT, 3>>> triples(n_available);
        vector<void*> supplies(n_available);
        for (int i = 0; i < n_available; i++)
        {
            supplies[i] = &triples[i];
            for (size_t j = 0;
                    j < n_per_thread * (B - 1) * n_bits_to_open / dl; j++)
                if (player < 0)
                    triples[i].push_back(proc.bit_prep.get_triple(dl));
                else
                    triples[i].push_back(personal_prep.get_triple(dl));
        }

        EdabitSacrificeJob job(&to_check, n_bits, strict, player);
        int start = queues->distribute_no_setup(job, N, 0, BT::default_length,
                &supplies);
        edabit_sacrifice_buckets(to_check, n_bits, strict, player, proc, start,
                N, personal_prep);
        if (start)
            queues->wrap_up(job);
    }
    else
        edabit_sacrifice_buckets(to_check, n_bits, strict, player, proc, 0, N,
                personal_prep);
#ifdef VERBOSE_EDA
    cerr << "Bucket sacrifice took " << bucket_timer.elapsed() << " seconds"
            << endl;
#endif

    RunningTimer output_timer;
    to_check.resize(N);
    output.reserve(output.size() + N);
    for (auto& x: to_check)
    {
        output.push_back({x.first, {}});
        output.back().second.reserve(x.second.size());
        for (auto& y : x.second)
            output.back().second.push_back(y);
    }
#ifdef VERBOSE_EDA
    cerr << "Output took " << output_timer.elapsed() << " seconds" << endl;
#endif

    MCB.Check(P);
    delete &MCB;

#ifdef VERBOSE_EDA
    cerr << "Done sacrificing edaBits of length " << n_bits << " after "
            << timer.elapsed() << " seconds" << endl;
#endif
}

template<class T>
void EdabitShuffleSacrifice<T>::edabit_sacrifice_buckets(vector<edabit<T>>& to_check,
        size_t n_bits, bool strict, int player, SubProcessor<T>& proc, int begin,
        int end, const void* supply)
{
    LimitedPrep<BT> personal_prep;
    edabit_sacrifice_buckets(to_check, n_bits, strict, player, proc, begin, end,
            personal_prep, supply);
}

template<class T>
void EdabitShuffleSacrifice<T>::edabit_sacrifice_buckets(vector<edabit<T>>& to_check,
        size_t n_bits, bool strict, int player, SubProcessor<T>& proc, int begin,
        int end, LimitedPrep<BT>& personal_prep, const void* supply)
{
    typedef typename T::bit_type::part_type BT;
    typedef typename BT::small_type ST;

    int N = end - begin;
    size_t total_N = to_check.size() / B;
    assert(to_check.size() == B * total_N);
    size_t n_bits_to_open = to_check[begin].second.size();
    ThreadQueues* queues = 0;
    auto& P = proc.P;
    auto& MC = proc.MC;

    // sacrifice buckets
    RunningTimer add_prep_timer;
    vector<vector<vector<BT>>> summands(n_bits_to_open,
            vector<vector<BT>>(2, vector<BT>((B - 1) * N / BT::default_length)));
    assert((B - 1) * N % BT::default_length == 0);
    for (int i = 0; i < N; i++)
    {
        auto& b = to_check[begin + i].second;
        assert(b.size() == n_bits_to_open);
        for (int j = 1; j < B; j++)
        {
            auto& g = to_check[begin + i + total_N * j].second;
            assert(g.size() == n_bits_to_open);
            int l = i * (B - 1) + j - 1;
            int l1 = l / BT::default_length;
            int l2 = l % BT::default_length;
            for (size_t k = 0; k < n_bits_to_open; k++)
            {
                summands[k][0].at(l1) ^= BT(b[k]) << (l2);
                summands[k][1].at(l1) ^= BT(g[k]) << (l2);
            }
        }
    }
#ifdef VERBOSE_EDA
    cerr << "Bit adder preparing took " << add_prep_timer.elapsed() <<
            " seconds" << endl;
#endif

    RunningTimer add_timer;
    vector<vector<BT>> sums;
    auto& party = GC::ShareThread<typename T::bit_type>::s();
    if (supply)
    {
        auto& triples = *(vector<array<BT, 3>>*)supply;
#ifdef VERBOSE_EDA
        fprintf(stderr, "got %zu supplies\n", triples.size());
#endif
        if (player < 0)
            proc.bit_prep.push_triples(triples);
        else
            personal_prep.push_triples(triples);
    }
    if (player < 0)
    {
        SubProcessor<BT> bit_proc(party.MC->get_part_MC(),
                proc.bit_prep, P);
        BitAdder().add(sums, summands, bit_proc, BT::default_length, queues);
    }
    else
    {
        SubProcessor<BT> bit_proc(party.MC->get_part_MC(),
                personal_prep, P);
        BitAdder().add(sums, summands, bit_proc, BT::default_length, queues,
                player);
    }
    summands.clear();
#ifdef VERBOSE_EDA
    cerr << "Binary adders took " << add_timer.elapsed() << " seconds" << endl;
#endif

    vector<T> shares;
    vector<typename BT::small_type> bit_shares;

    RunningTimer sacri_prep_timer;
    // cannot delete overflow in GF(p)
    strict |= T::open_type::N_BITS < 0;
    int n_shift = (T::open_type::N_BITS - n_bits);
    vector<edabit<T>> to_sanitize;
    if (strict)
        to_sanitize.reserve(N * (B - 1));
    else
    {
        shares.reserve((B - 1) * N);
        bit_shares.reserve((B - 1) * N * (n_bits + 2));
    }
    for (int i = 0; i < N; i++)
    {
        auto& a = to_check[begin + i].first;
        for (int j = 1; j < B; j++)
        {
            auto& f = to_check[begin + i + total_N * j].first;
            int l = i * (B - 1) + j - 1;
            int l1 = l / BT::default_length;
            int l2 = l % BT::default_length;
            typename edabit<T>::second_type* bits = 0;
            if (strict)
            {
                to_sanitize.push_back({a + f, {}});
                bits = &to_sanitize.back().second;
                bits->reserve(sums.at(l1).size());
            }
            else
            {
                shares.push_back((a + f) << n_shift);
            }
            for (size_t k = 0; k < (strict ? sums.at(l1).size() : n_bits); k++)
            {
                auto x = ST(sums.at(l1).at(k) >> l2).get_bit(0);
                if (strict)
                    bits->push_back(x);
                else
                    bit_shares.push_back(x);
            }
        }
    }

    if (strict)
    {
        (dynamic_cast<RingPrep<T>*>(&proc.DataF))->template
                sanitize<0>(to_sanitize,
                n_bits, player, queues);
        shares.reserve((B - 1) * N);
        bit_shares.reserve((B - 1) * N * (n_bits + 2));
        for (auto& x : to_sanitize)
        {
            shares.push_back(x.first);
            for (size_t k = 0; k < n_bits; k++)
                bit_shares.push_back(x.second[k]);
        }
        to_sanitize.clear();
        to_sanitize.shrink_to_fit();
    }

#ifdef VERBOSE_EDA
    cerr << "Preparing sacrifice took " << sacri_prep_timer.elapsed() <<
            " seconds" << endl;
    Timer open_timer;
    open_timer.start();
#endif
    vector<typename T::open_type> opened;
    vector<typename BT::small_type::open_type> bits;
    auto& MCB = *BT::small_type::new_mc(
            GC::ShareThread<typename T::bit_type>::s().MC->get_alphai());
    MC.POpen(opened, shares, P);
    MCB.POpen(bits, bit_shares, P);
#ifdef VERBOSE_EDA
    cerr << "Bucket opening took " << open_timer.elapsed() << " seconds" << endl;
#endif

    RunningTimer check_timer;
    for (int i = 0; i < (B - 1) * N; i++)
    {
        typename T::clear sum, single = opened[i];
        for (size_t k = 0; k < n_bits; k++)
            sum += T::clear::power_of_two(bits.at(i * (n_bits) + k).get_bit(0), k);
        if (not strict)
            sum <<= n_shift;
        if (single != sum)
        {
            cout << hex << single << " vs " << (sum << n_shift) << "/" << sum
                    << endl;
            throw Offline_Check_Error("edabit shuffle bucket opening");
        }
    }
#ifdef VERBOSE_EDA
    cerr << "Checking took " << check_timer.elapsed() << " seconds" << endl;
#endif

    MCB.Check(P);
    delete &MCB;
}

#endif /* PROTOCOLS_SHUFFLESACRIFICE_HPP_ */
