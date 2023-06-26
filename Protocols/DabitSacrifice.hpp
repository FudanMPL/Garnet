/*
 * DabitSacrifice.cpp
 *
 */

#ifndef PROTOCOLS_DABITSACRIFICE_HPP_
#define PROTOCOLS_DABITSACRIFICE_HPP_

#include "DabitSacrifice.h"
#include "Tools/PointerVector.h"

#include <math.h>

template<class T>
DabitSacrifice<T>::DabitSacrifice() :
        S(OnlineOptions::singleton.security_parameter)
{
}

template<class T>
dabit<T>& operator+=(dabit<T>& x, const dabit<T>& y)
{
    x.first += y.first;
    x.second ^= y.second;
    return x;
}

template<class T>
void DabitSacrifice<T>::sacrifice_without_bit_check(vector<dabit<T> >& dabits,
        vector<dabit<T> >& check_dabits, SubProcessor<T>& proc,
        ThreadQueues*)
{
#ifdef VERBOSE_DABIT
    cerr << "Sacrificing daBits" << endl;
    Timer timer;
    timer.start();
#endif
    int n = check_dabits.size() - S;
    GlobalPRNG G(proc.P);
    typedef typename T::bit_type::part_type BT;
    vector<T> shares;
    vector<BT> bit_shares;
    for (int i = 0; i < S; i++)
    {
        dabit<T> to_check;
        for (int j = 0; j < n; j++)
        {
            if (G.get_bit())
                to_check += check_dabits[j];
        }
        to_check += check_dabits[n + i];
        T masked = to_check.first;
        if (T::clear::N_BITS > 0)
            masked = masked << (T::clear::N_BITS - 1);
        else
            for (int j = 0; j < ceil(log2(n)) + S; j++)
            {
                T tmp;
                proc.DataF.get_one(DATA_BIT, tmp);
                masked += tmp << (1 + j);
            }
        shares.push_back(masked);
        bit_shares.push_back(to_check.second);
    }
    auto& MC = proc.MC;
    auto& MCBB = *BT::new_mc(
            GC::ShareThread<typename T::bit_type>::s().MC->get_alphai());
    vector<typename T::open_type> opened;
    vector<typename BT::open_type> bit_opened;
    MC.POpen(opened, shares, proc.P);
    MCBB.POpen(bit_opened, bit_shares, proc.P);
    for (int i = 0; i < S; i++)
    {
        auto a = typename T::clear(opened[i]);
        if (T::clear::N_BITS > 0)
            a >>= (T::clear::N_BITS - 1);
        else
            a &= 1;
        auto b = bit_opened[i];
        if (a != b.get())
        {
            cerr << a << " != " << b << endl;
            throw Offline_Check_Error("daBit sacrifice");
        }
    }
    dabits.insert(dabits.end(), check_dabits.begin(), check_dabits.begin() + n);
    MCBB.Check(proc.P);
    delete &MCBB;
#ifdef VERBOSE_DABIT
    cerr << "Done sacrificing daBits after " << timer.elapsed() << " seconds"
            << endl;
#endif
}

template<class T>
void DabitSacrifice<T>::sacrifice_and_check_bits(vector<dabit<T> >& dabits,
        vector<dabit<T> >& check_dabits, SubProcessor<T>& proc,
        ThreadQueues* queues)
{
    vector<dabit<T>> to_check;
    sacrifice_without_bit_check(to_check, check_dabits, proc, queues);
    typename T::Protocol protocol(proc.P);
    vector<pair<T, T>> multiplicands;
    for (auto& x : to_check)
        multiplicands.push_back({x.first, x.first});
    PointerVector<T> products(multiplicands.size());
    if (queues)
    {
        ThreadJob job(&products, &multiplicands);
        int start = queues->distribute(job, multiplicands.size());
        protocol.multiply(products, multiplicands, start, multiplicands.size(), proc);
        if (start)
            queues->wrap_up(job);
    }
    else
        protocol.multiply(products, multiplicands, 0, multiplicands.size(), proc);
    vector<T> check_for_zero;
    for (auto& x : to_check)
        check_for_zero.push_back(x.first - products.next());
    proc.MC.CheckFor(0, check_for_zero, proc.P);
    dabits.insert(dabits.end(), to_check.begin(), to_check.end());
}

#endif
