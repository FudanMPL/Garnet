/*
 * ThreadQueues.h
 *
 */

#ifndef PROCESSOR_THREADQUEUES_H_
#define PROCESSOR_THREADQUEUES_H_

#include "Tools/WaitQueue.h"
#include "ThreadJob.h"
#include "ThreadQueue.h"

class ThreadQueues :
        public vector<ThreadQueue*>
{
    vector<int> available;

public:
    int find_available();
    int get_n_per_thread(int n_items, int granularity = 1);
    // expects that the last slice is done by the caller
    int distribute(ThreadJob job, int n_items, int base = 0,
            int granularity = 1);
    int distribute_no_setup(ThreadJob job, int n_items, int base = 0,
            int granularity = 1, const vector<void*>* supplies = 0);
    void wrap_up(ThreadJob job);
};

#endif /* PROCESSOR_THREADQUEUES_H_ */
