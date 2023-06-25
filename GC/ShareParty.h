/*
 * ReplicatedParty.h
 *
 */

#ifndef GC_SHAREPARTY_H_
#define GC_SHAREPARTY_H_

#include "ShareSecret.h"
#include "Processor.h"
#include "Program.h"
#include "Memory.h"
#include "ThreadMaster.h"

namespace GC
{

template<class T>
class ShareParty : public ThreadMaster<T>
{
    static ShareParty<T>* singleton;

    ez::ezOptionParser& opt;
    OnlineOptions online_opts;

public:
    static ShareParty& s();

    typename T::mac_key_type mac_key;

    ShareParty(int argc, const char** argv, ez::ezOptionParser& opt,
            int default_batch_size = 0);

    Thread<T>* new_thread(int i);

    void post_run();
};

template<class T>
inline ShareParty<T>& ShareParty<T>::s()
{
    if (singleton)
        return *singleton;
    else
        throw runtime_error("no singleton");
}

}

#endif /* GC_SHAREPARTY_H_ */
