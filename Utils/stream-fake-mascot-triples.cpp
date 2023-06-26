/*
 * stream-fake-mascot-triples.cpp
 *
 */

#include "Protocols/Share.h"
#include "Math/gfpvar.h"
#include "Tools/benchmarking.h"

#include "Math/Setup.hpp"
#include "Protocols/fake-stuff.hpp"

class Info
{
public:
    int thread_num;
    int nplayers;
    gfpvar key;
    pthread_t thread;
};

void* run(void* arg)
{
    auto& info = *(Info*) arg;
    Files<Share<gfpvar>> files(info.nplayers, info.key, PREP_DIR, DATA_TRIPLE, info.thread_num);
    SeededPRNG G;
    int count = 0;
    while (true)
    {
        for (int i = 0; i < 100000; i++)
        {
            gfpvar triple[3];
            for (int i = 0; i < 2; i++)
                triple[i].randomize(G);
            triple[2] = triple[0] * triple[1];
            for (int i = 0; i < 3; i++)
                files.output_shares(triple[i]);
            count++;
        }
        // take a break to make them wait
        sleep(1);
    }
    cerr << "failed after " << count << endl;
    return 0;
}

int main()
{
    insecure_fake();
    typedef Share<gfpvar> T;
    int nplayers = 2;
    int lgp = 128;
    string prep_data_prefix = PREP_DIR;
    gfpvar::generate_setup<T>(prep_data_prefix, nplayers, lgp);
    T::mac_key_type keyp;
    generate_mac_keys<T>(keyp, nplayers, prep_data_prefix);

    int nthreads = 3;
    OnlineOptions::singleton.file_prep_per_thread = true;
    vector<Info> infos(3);
    for (int i = 0; i < nthreads; i++)
    {
        auto& info = infos[i];
        info.thread_num = i;
        info.nplayers = nplayers;
        info.key = keyp;
        pthread_create(&info.thread, 0, run, &info);
    }
    pthread_join(infos[0].thread, 0);
}
