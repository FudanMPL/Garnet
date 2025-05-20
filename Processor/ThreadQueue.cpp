/*
 * ThreadQueue.cpp
 *
 */


#include "ThreadQueue.h"

void ThreadQueue::schedule(const ThreadJob& job)
{
    lock.lock();
    left++;
#ifdef DEBUG_THREAD_QUEUE
        cerr << this << ": " << left << " left" << endl;
#endif
    lock.unlock();
    in.push(job);
}

ThreadJob ThreadQueue::next()
{
    return in.pop();
}

void ThreadQueue::finished(const ThreadJob& job)
{
    out.push(job);
}

void ThreadQueue::finished(const ThreadJob& job, const NamedCommStats& new_comm_stats)
{
    finished(job);
    set_comm_stats(new_comm_stats);
}

void ThreadQueue::set_comm_stats(const NamedCommStats& new_comm_stats)
{
    lock.lock();
    comm_stats = new_comm_stats;
    lock.unlock();
}

ThreadJob ThreadQueue::result()
{
    auto res = out.pop();
    lock.lock();
    left--;
#ifdef DEBUG_THREAD_QUEUE
        cerr << this << ": " << left << " left" << endl;
#endif
    lock.unlock();
    return res;
}

NamedCommStats ThreadQueue::get_comm_stats()
{
    lock.lock();
    auto res = comm_stats;
    lock.unlock();
    return res;
}
