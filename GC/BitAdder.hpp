/*
 * BitAdder.cpp
 *
 */

#ifndef GC_BITADDER_HPP_
#define GC_BITADDER_HPP_

#include "BitAdder.h"

#include <assert.h>

template<class T>
void BitAdder::add(vector<vector<T>>& res, const vector<vector<vector<T>>>& summands,
        SubProcessor<T>& proc, int length, ThreadQueues* queues, int player)
{
    assert(not summands.empty());
    assert(not summands[0].empty());

    res.resize(summands[0][0].size());

    if (queues)
    {
        assert(length == T::default_length);
        int n_available = queues->find_available();
        int n_per_thread = queues->get_n_per_thread(res.size());
        vector<vector<array<T, 3>>> triples(n_available);
        vector<void*> supplies(n_available);
        for (int i = 0; i < n_available; i++)
        {
            if (T::expensive_triples)
            {
                supplies[i] = &triples[i];
                for (size_t j = 0; j < n_per_thread * summands.size(); j++)
                    triples[i].push_back(proc.DataF.get_triple(T::default_length));
#ifdef VERBOSE_EDA
                cerr << "supplied " << triples[i].size() << endl;
#endif
            }
        }

        ThreadJob job(&res, &summands, T::default_length, player);
        int start = queues->distribute_no_setup(job, res.size(), 0, 1,
                &supplies);
        BitAdder().add(res, summands, start,
                summands[0][0].size(), proc, T::default_length);
        if (start)
            queues->wrap_up(job);
    }
    else
        add(res, summands, 0, res.size(), proc, length);
}

template<class T>
void BitAdder::add(vector<vector<T> >& res,
        const vector<vector<vector<T> > >& summands, size_t begin, size_t end,
        SubProcessor<T>& proc, int length, int input_begin, const void* supply)
{
#ifdef VERBOSE_EDA
    fprintf(stderr, "add bits %lu to %lu\n", begin, end);
#endif

    if (input_begin < 0)
        input_begin = begin;

    int n_bits = summands.size();
    for (size_t i = begin; i < end; i++)
        res.at(i).resize(n_bits + 1);

    size_t n_items = end - begin;

    if (supply)
    {
#ifdef VERBOSE_EDA
        fprintf(stderr, "got supply\n");
#endif
        auto& s = *(vector<array<T, 3>>*) supply;
        assert(s.size() == n_items * n_bits);
        proc.DataF.push_triples(s);
    }

    if (summands[0].size() > 2)
        return multi_add(res, summands, begin, end, proc, length, input_begin);

    vector<T> carries(n_items);
    vector<T> a(n_items), b(n_items);
    auto& protocol = proc.protocol;
    for (int i = 0; i < n_bits; i++)
    {
        assert(summands[i].size() == 2);
        assert(summands[i][0].size() >= input_begin + n_items);
        assert(summands[i][1].size() >= input_begin + n_items);

        for (size_t j = 0; j < n_items; j++)
        {
            a[j] = summands[i][0][input_begin + j];
            b[j] = summands[i][1][input_begin + j];
        }

        protocol.init_mul();
        for (size_t j = 0; j < n_items; j++)
        {
            res[begin + j][i] = a[j] + b[j] + carries[j];
            // full adder using MUX
            protocol.prepare_mul(a[j] + b[j], carries[j] + a[j], length);
        }
        protocol.exchange();
        for (size_t j = 0; j < n_items; j++)
            carries[j] = a[j] + protocol.finalize_mul(length);
    }

    for (size_t j = 0; j < n_items; j++)
        res[begin + j][n_bits] = carries[j];
}

template<class T>
void BitAdder::multi_add(vector<vector<T> >& res,
        const vector<vector<vector<T> > >& summands, size_t begin, size_t end,
        SubProcessor<T>& proc, int length, int input_begin)
{
    int n_bits = summands.size() + ceil(log2(proc.P.num_players()));
    size_t n_items = end - begin;

    assert(n_bits > 0);

    vector<vector<vector<T>>> my_summands(n_bits);

    for (auto& x : my_summands)
    {
        x.resize(2);
        for (auto& y : x)
            y.resize(n_items);
    }

    for (size_t i = 0; i < summands.size(); i++)
        for (int j = 0; j < 2; j++)
        {
            auto& x = my_summands.at(i).at(j);
            auto& z = summands.at(i).at(j);
            auto y = z.begin() + input_begin;
            assert(y + n_items <= z.end());
            x.clear();
            x.insert(x.begin(), y, y + n_items);
        }

    vector<vector<T>> my_res(n_items);
    for (size_t k = 2; k < summands.at(0).size(); k++)
    {
        add(my_res, my_summands, 0, n_items, proc, length);

        for (size_t i = 0; i < my_summands.size(); i++)
            for (size_t j = 0; j < n_items; j++)
                my_summands[i][0][j] = my_res[j][i];

        for (size_t i = 0; i < summands.size(); i++)
        {
            auto& z = summands.at(i).at(k);
            auto y = z.begin() + input_begin;
            assert(y + n_items <= z.end());
            auto& x = my_summands.at(i).at(1);
            x.clear();
            x.insert(x.begin(), y, y + n_items);
        }
    }

    add(res, my_summands, begin, end, proc, length, 0);
}

#endif
