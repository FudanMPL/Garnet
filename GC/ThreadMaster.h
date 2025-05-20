/*
 * ThreadMaster.h
 *
 */

#ifndef GC_THREADMASTER_H_
#define GC_THREADMASTER_H_

#include "Thread.h"
#include "Program.h"

#include "Processor/OnlineOptions.h"

namespace GC
{

class ThreadMasterBase
{
public:
    Names N;
    string progname;

    virtual ~ThreadMasterBase() {}

    virtual void run()
    {
        throw runtime_error("cannot run base class");
    }

    void run(string& progname)
    {
        this->progname = progname;
        run();
    }
};

template<class T>
class ThreadMaster : public ThreadMasterBase
{
    static ThreadMaster<T>* singleton;

public:
    vector<Thread<T>*> threads;

    Player* P;

    Machine<T> machine;
    typename T::DynamicMemory memory;

    OnlineOptions& opts;

    static ThreadMaster<T>& s();

    ThreadMaster(OnlineOptions& opts);
    virtual ~ThreadMaster() {}

    void run_tape(int thread_number, int tape_number, int arg);
    void join_tape(int thread_number);

    virtual Thread<T>* new_thread(int i);

    void run();

    virtual void post_run() {}
};

} /* namespace GC */

#endif /* GC_THREADMASTER_H_ */
