/*
 * BaseMachine.h
 *
 */

#ifndef PROCESSOR_BASEMACHINE_H_
#define PROCESSOR_BASEMACHINE_H_

#include "Tools/time-func.h"
#include "Tools/TimerWithComm.h"
#include "OT/OTTripleSetup.h"
#include "ThreadJob.h"
#include "ThreadQueues.h"

#include <map>
#include <fstream>
using namespace std;

void print_usage(ostream& o, const char* name, size_t capacity);

class BaseMachine
{
protected:
    static BaseMachine* singleton;

    static thread_local OnDemandOTTripleSetup ot_setup;

    std::map<int,TimerWithComm> timer;

    string compiler;
    string domain;
    string relevant_opts;

    virtual size_t load_program(const string& threadname,
            const string& filename);

public:
    static thread_local int thread_num;

    string progname;
    int nthreads;

    ThreadQueues queues;

    vector<string> bc_filenames;

    static BaseMachine& s();
    static bool has_singleton() { return singleton != 0; }

    static string memory_filename(const string& type_short, int my_number);

    static string get_domain(string progname);
    static int ring_size_from_schedule(string progname);
    static int prime_length_from_schedule(string progname);
    static bigint prime_from_schedule(string progname);

    BaseMachine();
    virtual ~BaseMachine() {}

    void load_schedule(const string& progname, bool load_bytecode = true);
    void print_compiler();

    void time();
    void start(int n);
    void stop(int n);

    void print_timers();

    virtual void reqbl(int) {}

    static OTTripleSetup fresh_ot_setup(Player& P);

    NamedCommStats total_comm();
    void set_thread_comm(const NamedCommStats& stats);

    void print_global_comm(Player& P, const NamedCommStats& stats);
};

inline OTTripleSetup BaseMachine::fresh_ot_setup(Player& P)
{
    return ot_setup.get_fresh(P);
}

#endif /* PROCESSOR_BASEMACHINE_H_ */
