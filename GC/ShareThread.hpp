/*
 * MalicousRepParty.cpp
 *
 */

#ifndef GC_SHARETHREAD_HPP_
#define GC_SHARETHREAD_HPP_

#include <GC/ShareThread.h>
#include "GC/ShareParty.h"
#include "BitPrepFiles.h"
#include "Math/Setup.h"

#include "Processor/Data_Files.hpp"

namespace GC
{

template<class T>
StandaloneShareThread<T>::StandaloneShareThread(int i, ThreadMaster<T>& master) :
        ShareThread<T>(*Preprocessing<T>::get_new(master.opts.live_prep,
                master.N, usage)),
        Thread<T>(i, master)
{
}

template<class T>
StandaloneShareThread<T>::~StandaloneShareThread()
{
    delete &this->DataF;
}

template<class T>
ShareThread<T>::ShareThread(Preprocessing<T>& prep) :
    P(0), MC(0), protocol(0), DataF(prep)
{
}

template<class T>
ShareThread<T>::ShareThread(Preprocessing<T>& prep, Player& P,
        typename T::mac_key_type mac_key) :
        ShareThread(prep)
{
    pre_run(P, mac_key);
}

template<class T>
ShareThread<T>::~ShareThread()
{
    if (MC)
        delete MC;
    if (protocol)
        delete protocol;
}

template<class T>
void ShareThread<T>::pre_run(Player& P, typename T::mac_key_type mac_key)
{
    this->P = &P;
    if (singleton)
        throw runtime_error("there can only be one");
    singleton = this;
    protocol = new typename T::Protocol(*this->P);
    MC = this->new_mc(mac_key);
    DataF.set_protocol(*this->protocol);
    this->protocol->init(DataF, *MC);
}

template<class T>
void StandaloneShareThread<T>::pre_run()
{
    ShareThread<T>::pre_run(*Thread<T>::P, ShareParty<T>::s().mac_key);
    usage.set_num_players(Thread<T>::P->num_players());
}

template<class T>
void ShareThread<T>::post_run()
{
    check();
}

template<class T>
void ShareThread<T>::check()
{
    protocol->check();
    MC->Check(*this->P);
}

template<class T>
void ShareThread<T>::and_(Processor<T>& processor,
        const vector<int>& args, bool repeat)
{
    auto& protocol = this->protocol;
    processor.check_args(args, 4);
    protocol->init_mul();
    T x_ext, y_ext;
    for (size_t i = 0; i < args.size(); i += 4)
    {
        int n_bits = args[i];
        int left = args[i + 2];
        int right = args[i + 3];
        for (int j = 0; j < DIV_CEIL(n_bits, T::default_length); j++)
        {
            int n = min(T::default_length, n_bits - j * T::default_length);
            if (repeat)
                processor.S[right].extend_bit(y_ext, n);
            else
                processor.S[right + j].mask(y_ext, n);
            processor.S[left + j].mask(x_ext, n);
            protocol->prepare_mult(x_ext, y_ext, n, repeat);
        }
    }

    protocol->exchange();

    for (size_t i = 0; i < args.size(); i += 4)
    {
        int n_bits = args[i];
        int out = args[i + 1];
        for (int j = 0; j < DIV_CEIL(n_bits, T::default_length); j++)
        {
            int n = min(T::default_length, n_bits - j * T::default_length);
            auto& res = processor.S[out + j];
            protocol->finalize_mult(res, n);
            res.mask(res, n);
        }
    }
}

template<class T>
void ShareThread<T>::andrsvec(Processor<T>& processor, const vector<int>& args)
{
    int N_BITS = T::default_length;
    auto& protocol = this->protocol;
    assert(protocol);
    protocol->init_mul();
    auto it = args.begin();
    T x_ext, y_ext;
    while (it < args.end())
    {
        int n_args = (*it++ - 3) / 2;
        int size = *it++;
        it += n_args;
        int base = *it++;
        assert(n_args <= N_BITS);
        for (int i = 0; i < size; i += N_BITS)
        {
            int n_ops = min(N_BITS, size - i);
            for (int j = 0; j < n_args; j++)
            {
                processor.S.at(*(it + j) + i / N_BITS).mask(x_ext, n_ops);
                processor.S.at(base + i / N_BITS).mask(y_ext, n_ops);
                protocol->prepare_mul(x_ext, y_ext, n_ops);
            }
        }
        it += n_args;
    }

    protocol->exchange();

    it = args.begin();
    while (it < args.end())
    {
        int n_args = (*it++ - 3) / 2;
        int size = *it++;
        for (int i = 0; i < size; i += N_BITS)
        {
            int n_ops = min(N_BITS, size - i);
            for (int j = 0; j < n_args; j++)
                protocol->finalize_mul(n_ops).mask(
                        processor.S.at(*(it + j) + i / N_BITS), n_ops);
        }
        it += 2 * n_args + 1;
    }
}

template<class T>
void ShareThread<T>::xors(Processor<T>& processor, const vector<int>& args)
{
    processor.check_args(args, 4);
    for (size_t i = 0; i < args.size(); i += 4)
    {
        int n_bits = args[i];
        int out = args[i + 1];
        int left = args[i + 2];
        int right = args[i + 3];
        for (int j = 0; j < DIV_CEIL(n_bits, T::default_length); j++)
        {
            int n = min(T::default_length, n_bits - j * T::default_length);
            processor.S[out + j].xor_(n, processor.S[left + j],
                    processor.S[right + j]);
        }
    }
}

} /* namespace GC */

#endif
