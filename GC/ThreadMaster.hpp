/*
 * ThreadMaster.cpp
 *
 */

#ifndef GC_THREADMASTER_HPP_
#define GC_THREADMASTER_HPP_

#include "ThreadMaster.h"
#include "Program.h"

#include "instructions.h"

#include "Tools/benchmarking.h"

#include "Machine.hpp"

namespace GC
{

template<class T>
ThreadMaster<T>* ThreadMaster<T>::singleton = 0;

template<class T>
ThreadMaster<T>& ThreadMaster<T>::s()
{
    if (singleton)
        return *singleton;
    else
        throw no_singleton("no singleton, maybe threads not supported");
}

template<class T>
ThreadMaster<T>::ThreadMaster(OnlineOptions& opts) :
        P(0), opts(opts)
{
    if (singleton)
        throw runtime_error("there can only be one");
    singleton = this;
}

template<class T>
void ThreadMaster<T>::run_tape(int thread_number, int tape_number, int arg)
{
    threads.at(thread_number)->tape_schedule.push({tape_number, arg});
}

template<class T>
void ThreadMaster<T>::join_tape(int thread_number)
{
    threads.at(thread_number)->join_tape();
}

template<class T>
Thread<T>* ThreadMaster<T>::new_thread(int i)
{
    return new Thread<T>(i, *this);
}

template<class T>
void ThreadMaster<T>::run()
{
    if (not opts.live_prep)
    {
        insecure("preprocessing from file in binary virtual machines");
    }

    P = new PlainPlayer(N, "main");

    machine.load_schedule(progname);
    machine.reset(machine.progs[0], memory);

    for (int i = 0; i < machine.nthreads; i++)
        threads.push_back(new_thread(i));
    for (auto thread : threads)
        thread->join_tape();

    machine.reset_timer();

    threads[0]->tape_schedule.push(0);

    for (auto thread : threads)
        thread->finish();

    // synchronize
    vector<octetStream> os(P->num_players());
    P->Broadcast_Receive(os);

    post_run();

    NamedCommStats stats = P->total_comm();
    ExecutionStats exe_stats;
    for (auto thread : threads)
    {
        stats += thread->P->total_comm();
        exe_stats += thread->processor.stats;
        delete thread;
    }

    exe_stats.print();
    stats.print();

    machine.print_timers();

    cerr << "Data sent = " << stats.sent * 1e-6 << " MB" << endl;

    machine.print_global_comm(*P, stats);

    delete P;
}

} /* namespace GC */

#endif
