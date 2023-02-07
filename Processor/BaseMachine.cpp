/*
 * BaseMachine.cpp
 *
 */

#include "BaseMachine.h"
#include "OnlineOptions.h"
#include "Math/Setup.h"
#include "Tools/Bundle.h"

#include <iostream>
#include <sodium.h>
using namespace std;

BaseMachine* BaseMachine::singleton = 0;
thread_local int BaseMachine::thread_num;
thread_local OnDemandOTTripleSetup BaseMachine::ot_setup;

void print_usage(ostream& o, const char* name, size_t capacity)
{
  if (capacity)
    o << name << "=" << capacity << " ";
}

BaseMachine& BaseMachine::s()
{
  if (singleton)
    return *singleton;
  else
    throw runtime_error("no singleton");
}

BaseMachine::BaseMachine() : nthreads(0)
{
  if (sodium_init() == -1)
    throw runtime_error("couldn't initialize libsodium");
  if (not singleton)
    singleton = this;
}

void BaseMachine::load_schedule(const string& progname, bool load_bytecode)
{
  this->progname = progname;
  string fname = "Programs/Schedules/" + progname + ".sch";
#ifdef DEBUG_FILES
  cerr << "Opening file " << fname << endl;
#endif
  ifstream inpf;
  inpf.open(fname);
  if (inpf.fail()) { throw file_error("Missing '" + fname + "'. Did you compile '" + progname + "'?"); }

  int nprogs;
  inpf >> nthreads;
  inpf >> nprogs;

  if (inpf.fail())
    throw file_error("Error reading " + fname);

#ifdef DEBUG_FILES
  cerr << "Number of threads I will run in parallel = " << nthreads << endl;
  cerr << "Number of program sequences I need to load = " << nprogs << endl;
#endif

  bc_filenames.clear();

  // Load in the programs
  string threadname;
  for (int i=0; i<nprogs; i++)
    { inpf >> threadname;
      size_t split = threadname.find(":");
      long expected = -1;
      if (split != string::npos)
        {
          expected = atoi(threadname.substr(split + 1).c_str());
          threadname = threadname.substr(0, split);
        }

      string filename = "Programs/Bytecode/" + threadname + ".bc";
      bc_filenames.push_back(filename);
      if (load_bytecode)
        {
#ifdef DEBUG_FILES
          cerr << "Loading program " << i << " from " << filename << endl;
#endif
          long size = load_program(threadname, filename);
          if (expected >= 0 and expected != size)
            throw runtime_error("broken bytecode file");
        }

    }

  for (auto i : {1, 0, 0})
    {
      int n;
      inpf >> n;
      if (n != i)
        throw runtime_error("old schedule format not supported");
    }

  inpf.get();
  getline(inpf, compiler);
  getline(inpf, domain);
  getline(inpf, relevant_opts);
  inpf.close();
}

void BaseMachine::print_compiler()
{
  if (compiler.size() != 0 and OnlineOptions::singleton.verbose)
    cerr << "Compiler: " << compiler << endl;
}

size_t BaseMachine::load_program(const string& threadname,
    const string& filename)
{
  (void)threadname;
  (void)filename;
  throw not_implemented();
}

void BaseMachine::time()
{
  cout << "Elapsed time: " << timer[0].elapsed() << endl;
}

void BaseMachine::start(int n)
{
  cout << "Starting timer " << n << " at " << timer[n].elapsed()
    << " (" << timer[n].mb_sent() << " MB)"
    << " after " << timer[n].idle() << endl;
  timer[n].start(total_comm());
}

void BaseMachine::stop(int n)
{
  timer[n].stop(total_comm());
  cout << "Stopped timer " << n << " at " << timer[n].elapsed() << " ("
      << timer[n].mb_sent() << " MB)" << endl;
}

void BaseMachine::print_timers()
{
  cerr << "The following benchmarks are ";
  if (OnlineOptions::singleton.live_prep)
    cerr << "in";
  else
    cerr << "ex";
  cerr << "cluding preprocessing (offline phase)." << endl;
  cerr << "Time = " << timer[0].elapsed() << " seconds " << endl;
  timer.erase(0);
  for (auto it = timer.begin(); it != timer.end(); it++)
    cerr << "Time" << it->first << " = " << it->second.elapsed() << " seconds ("
        << it->second.mb_sent() << " MB)" << endl;
}

string BaseMachine::memory_filename(const string& type_short, int my_number)
{
  return PREP_DIR "Memory-" + type_short + "-P" + to_string(my_number);
}

string BaseMachine::get_domain(string progname)
{
  assert(not singleton);
  BaseMachine machine;
  singleton = 0;
  machine.load_schedule(progname, false);
  return machine.domain;
}

int BaseMachine::ring_size_from_schedule(string progname)
{
  string domain = get_domain(progname);
  if (domain.substr(0, 2).compare("R:") == 0)
  {
    return stoi(domain.substr(2));
  }
  else
    return 0;
}

int BaseMachine::prime_length_from_schedule(string progname)
{
  string domain = get_domain(progname);
  if (domain.substr(0, 4).compare("lgp:") == 0)
    return stoi(domain.substr(4));
  else
    return 0;
}

bigint BaseMachine::prime_from_schedule(string progname)
{
  string domain = get_domain(progname);
  if (domain.substr(0, 2).compare("p:") == 0)
    return bigint(domain.substr(2));
  else
    return 0;
}

NamedCommStats BaseMachine::total_comm()
{
  NamedCommStats res;
  for (auto& queue : queues)
    res += queue->get_comm_stats();
  return res;
}

void BaseMachine::set_thread_comm(const NamedCommStats& stats)
{
  auto queue = queues.at(BaseMachine::thread_num);
  assert(queue);
  queue->set_comm_stats(stats);
}

void BaseMachine::print_global_comm(Player& P, const NamedCommStats& stats)
{
  Bundle<octetStream> bundle(P);
  bundle.mine.store(stats.sent);
  P.Broadcast_Receive_no_stats(bundle);
  size_t global = 0;
  for (auto& os : bundle)
    global += os.get_int(8);
  cerr << "Global data sent = " << global / 1e6 << " MB (all parties)" << endl;
}
