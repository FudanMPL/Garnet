#ifndef _Data_Files
#define _Data_Files

/* This class holds the Online data files all in one place
 * so the streams are easy to pass around and access
 */

#include "Math/field_types.h"
#include "Tools/Buffer.h"
#include "Processor/InputTuple.h"
#include "Tools/Lock.h"
#include "Networking/Player.h"
#include "Protocols/edabit.h"
#include "PrepBase.h"

#include <fstream>
#include <map>
using namespace std;

template<class T> class dabit;

namespace GC
{
template<class T> class ShareThread;
}

class DataTag
{
  int t[4];

public:
  // assume that tag is three integers
  DataTag(const int* tag)
  {
    strncpy((char*)t, (char*)tag, 3 * sizeof(int));
    t[3] = 0;
  }
  string get_string() const
  {
    return string((char*)t);
  }
  bool operator<(const DataTag& other) const
  {
    for (int i = 0; i < 3; i++)
      if (t[i] != other.t[i])
        return t[i] < other.t[i];
    return false;
  }
};

class DataPositions
{
  void process_line(long long items_used, const char* name, ifstream& file,
      bool print_verbose, double& total_cost, bool& reading_field,
      string suffix = "") const;

public:
  static const char* dtype_names[];
  static const char* field_names[N_DATA_FIELD_TYPE];
  static const int tuple_size[N_DTYPE];

  array<array<long long, N_DTYPE>, N_DATA_FIELD_TYPE> files;
  vector< array<long long, N_DATA_FIELD_TYPE> > inputs;
  array<map<DataTag, long long>, N_DATA_FIELD_TYPE> extended;
  map<pair<bool, int>, long long> edabits;
  map<array<int, 3>, long long> matmuls;

  DataPositions(int num_players = 0);
  DataPositions(const Player& P) : DataPositions(P.num_players()) {}
  ~DataPositions();

  void reset();
  void set_num_players(int num_players);
  int num_players() { return inputs.size(); }

  void count(DataFieldType type, DataTag tag, int n = 1);
  void count_edabit(bool strict, int n_bits);

  void increase(const DataPositions& delta);
  DataPositions& operator-=(const DataPositions& delta);
  DataPositions operator-(const DataPositions& delta) const;
  DataPositions operator+(const DataPositions& delta) const;
  void print_cost() const;
  bool empty() const;
  bool any_more(const DataPositions& other) const;
};

template<class sint, class sgf2n> class Processor;
template<class sint, class sgf2n> class Data_Files;
template<class sint, class sgf2n> class Machine;
template<class T> class SubProcessor;
template<class T> class NoFilePrep;

/**
 * Abstract base class for preprocessing
 */
template<class T>
class Preprocessing : public PrepBase
{
protected:
  static const bool use_part = false;

  DataPositions& usage;

  map<pair<bool, int>, vector<edabitvec<T>>> edabits;
  map<pair<bool, int>, edabitvec<T>> my_edabits;

  bool do_count;

  void count(Dtype dtype, int n = 1)
  { usage.files[T::clear::field_type()][dtype] += do_count * n; }
  void count_input(int player)
  { usage.inputs[player][T::clear::field_type()] += do_count; }

  template<int>
  void get_edabits(bool strict, size_t size, T* a,
      vector<typename T::bit_type>& Sb, const vector<int>& regs, false_type);
  template<int>
  void get_edabits(bool, size_t, T*, vector<typename T::bit_type>&,
      const vector<int>&, true_type)
  { throw not_implemented(); }

  T get_random_from_inputs(int nplayers);

public:
  template<class U, class V>
  static Preprocessing<T>* get_new(Machine<U, V>& machine, DataPositions& usage,
      SubProcessor<T>* proc);
  template<int = 0>
  static Preprocessing<T>* get_new(bool live_prep, const Names& N,
      DataPositions& usage);
  static Preprocessing<T>* get_live_prep(SubProcessor<T>* proc,
      DataPositions& usage);

  Preprocessing(DataPositions& usage) : usage(usage), do_count(true) {}
  virtual ~Preprocessing() {}

  virtual void set_protocol(typename T::Protocol&) {};
  virtual void set_proc(SubProcessor<T>* proc) { (void) proc; }

  virtual void seekg(DataPositions& pos) { (void) pos; }
  virtual void prune() {}
  virtual void purge() {}

  virtual void get_three_no_count(Dtype, T&, T&, T&)
  { throw not_implemented(); }
  virtual void get_two_no_count(Dtype, T&, T&) { throw not_implemented(); }
  virtual void get_one_no_count(Dtype, T&) { throw not_implemented(); }
  virtual void get_input_no_count(T&, typename T::open_type&, int)
  { throw not_implemented() ; }
  virtual void get_no_count(vector<T>&, DataTag, const vector<int>&, int)
  { throw not_implemented(); }

  void get(Dtype dtype, T* a);
  void get_three(Dtype dtype, T& a, T& b, T& c);
  void get_two(Dtype dtype, T& a, T& b);
  void get_one(Dtype dtype, T& a);
  void get_input(T& a, typename T::open_type& x, int i);
  void get(vector<T>& S, DataTag tag, const vector<int>& regs, int vector_size);

  /// Get fresh random multiplication triple
  virtual array<T, 3> get_triple(int n_bits);
  virtual array<T, 3> get_triple_no_count(int n_bits);
  /// Get fresh random bit
  virtual T get_bit();
  /// Get fresh random value in domain
  virtual T get_random();
  /// Store fresh daBit in ``a`` (arithmetic part) and ``b`` (binary part)
  virtual void get_dabit(T& a, typename T::bit_type& b);
  virtual void get_dabit_no_count(T&, typename T::bit_type&) { throw runtime_error("no daBit"); }
  virtual void get_edabits(bool strict, size_t size, T* a,
          vector<typename T::bit_type>& Sb, const vector<int>& regs)
  { get_edabits<0>(strict, size, a, Sb, regs, T::clear::characteristic_two); }
  template<int>
  void get_edabit_no_count(bool, int n_bits, edabit<T>& eb);
  template<int>
  /// Get fresh edaBit chunk
  edabitvec<T> get_edabitvec(bool strict, int n_bits);
  virtual void buffer_edabits_with_queues(bool, int) { throw runtime_error("no edaBits"); }

  virtual void push_triples(const vector<array<T, 3>>&)
  { throw runtime_error("no pushing"); }

  virtual void buffer_triples() {}
  virtual void buffer_inverses() {}

  virtual Preprocessing<typename T::part_type>& get_part() { throw runtime_error("no part"); }
};

template<class T>
class Sub_Data_Files : public Preprocessing<T>
{
  template<class U> friend class Sub_Data_Files;

  typedef typename conditional<T::LivePrep::use_part,
      Sub_Data_Files<typename T::part_type>, NoFilePrep<typename T::part_type>>::type part_type;

  static int tuple_length(int dtype);

  BufferOwner<T, T> buffers[N_DTYPE];
  vector<BufferOwner<T, T>> input_buffers;
  BufferOwner<InputTuple<T>, RefInputTuple<T>> my_input_buffers;
  map<DataTag, BufferOwner<T, T> > extended;
  BufferOwner<dabit<T>, dabit<T>> dabit_buffer;
  map<int, ifstream*> edabit_buffers;

  int my_num,num_players;

  const string prep_data_dir;
  int thread_num;

  part_type* part;

  void buffer_edabits_with_queues(bool strict, int n_bits)
  { buffer_edabits_with_queues<0>(strict, n_bits, T::clear::characteristic_two); }
  template<int>
  void buffer_edabits_with_queues(bool strict, int n_bits, false_type);
  template<int>
  void buffer_edabits_with_queues(bool, int, true_type)
  { throw not_implemented(); }

public:
  static string get_filename(const Names& N, Dtype type, int thread_num = -1);
  static string get_input_filename(const Names& N, int input_player,
      int thread_num = -1);
  static string get_edabit_filename(const Names& N, int n_bits,
      int thread_num = -1);

  Sub_Data_Files(int my_num, int num_players, const string& prep_data_dir,
      DataPositions& usage, int thread_num = -1);
  Sub_Data_Files(const Names& N, DataPositions& usage, int thread_num = -1);
  Sub_Data_Files(const Names& N, const string& prep_data_dir,
      DataPositions& usage, int thread_num = -1) :
      Sub_Data_Files(N.my_num(), N.num_players(), prep_data_dir, usage, thread_num)
  {
  }
  ~Sub_Data_Files();

  void set_protocol(typename T::Protocol& protocol) { (void) protocol; }

  void seekg(DataPositions& pos);
  void prune();
  void purge();

  bool eof(Dtype dtype);
  bool input_eof(int player);

  void get_no_count(Dtype dtype, T* a);

  void get_three_no_count(Dtype dtype, T& a, T& b, T& c)
  {
    buffers[dtype].input(a);
    buffers[dtype].input(b);
    buffers[dtype].input(c);
  }

  void get_two_no_count(Dtype dtype, T& a, T& b)
  {
    buffers[dtype].input(a);
    buffers[dtype].input(b);
  }

  void get_one_no_count(Dtype dtype, T& a)
  {
    buffers[dtype].input(a);
  }

  void get_input_no_count(T& a,typename T::open_type& x,int i)
  {
    RefInputTuple<T> tuple(a, x);
    if (i==my_num)
      my_input_buffers.input(tuple);
    else
      input_buffers[i].input(a);
  }

  void setup_extended(const DataTag& tag, int tuple_size = 0);
  void get_no_count(vector<T>& S, DataTag tag, const vector<int>& regs, int vector_size);
  void get_dabit_no_count(T& a, typename T::bit_type& b);

  part_type& get_part();
};

template<class sint, class sgf2n>
class Data_Files
{
  friend class Processor<sint, sgf2n>;

  DataPositions usage, skipped;

  public:

  Preprocessing<sint>& DataFp;
  Preprocessing<sgf2n>& DataF2;
  Preprocessing<typename sint::bit_type>& DataFb;

  Data_Files(Machine<sint, sgf2n>& machine, SubProcessor<sint>* procp = 0,
      SubProcessor<sgf2n>* proc2 = 0);
  Data_Files(const Names& N);
  ~Data_Files();

  DataPositions tellg() { return usage; }
  void seekg(DataPositions& pos);
  void skip(const DataPositions& pos);
  void prune();
  void purge();

  DataPositions get_usage()
  {
    return usage - skipped;
  }

  void reset_usage() { usage.reset(); skipped.reset(); }

  void set_usage(const DataPositions& pos) { usage = pos; }
};

template<class T> inline
bool Sub_Data_Files<T>::eof(Dtype dtype)
  { return buffers[dtype].eof; }

template<class T> inline
bool Sub_Data_Files<T>::input_eof(int player)
{
  if (player == my_num)
    return my_input_buffers.eof;
  else
    return input_buffers[player].eof;
}

template<class T>
inline void Sub_Data_Files<T>::get_no_count(Dtype dtype, T* a)
{
  for (int i = 0; i < DataPositions::tuple_size[dtype]; i++)
    buffers[dtype].input(a[i]);
}

template<class T>
inline void Preprocessing<T>::get(Dtype dtype, T* a)
{
  switch (dtype)
  {
  case DATA_TRIPLE:
      get_three(dtype, a[0], a[1], a[2]);
      break;
  case DATA_SQUARE:
  case DATA_INVERSE:
      get_two(dtype, a[0], a[1]);
      break;
  case DATA_BIT:
      get_one(dtype, a[0]);
      break;
  default:
      throw runtime_error("unsupported data type: " + to_string(dtype));
  }
}

template<class T>
inline void Preprocessing<T>::get_three(Dtype dtype, T& a, T& b, T& c)
{
  // count bit triples in get_triple()
  if (T::clear::field_type() != DATA_GF2)
    count(dtype);
  get_three_no_count(dtype, a, b, c);
}

template<class T>
inline void Preprocessing<T>::get_two(Dtype dtype, T& a, T& b)
{
  count(dtype);
  get_two_no_count(dtype, a, b);
}

template<class T>
inline void Preprocessing<T>::get_one(Dtype dtype, T& a)
{
  count(dtype);
  get_one_no_count(dtype, a);
}

template<class T>
inline void Preprocessing<T>::get_input(T& a, typename T::open_type& x, int i)
{
  count_input(i);
  get_input_no_count(a, x, i);
}

template<class T>
inline void Preprocessing<T>::get(vector<T>& S, DataTag tag,
    const vector<int>& regs, int vector_size)
{
  usage.count(T::clear::field_type(), tag, vector_size);
  get_no_count(S, tag, regs, vector_size);
}

template<class T>
array<T, 3> Preprocessing<T>::get_triple(int n_bits)
{
  if (T::clear::field_type() == DATA_GF2)
    count(DATA_TRIPLE, n_bits);
  return get_triple_no_count(n_bits);
}

template<class T>
array<T, 3> Preprocessing<T>::get_triple_no_count(int n_bits)
{
  assert(T::clear::field_type() != DATA_GF2 or T::default_length == 1 or
      T::default_length == n_bits or not do_count);
  array<T, 3> res;
  get(DATA_TRIPLE, res.data());
  return res;
}

template<class T>
T Preprocessing<T>::get_bit()
{
  T res;
  get_one(DATA_BIT, res);
  return res;
}

template<class T>
T Preprocessing<T>::get_random()
{
  assert(not usage.inputs.empty());
  return get_random_from_inputs(usage.inputs.size());
}

template<class sint, class sgf2n>
inline void Data_Files<sint, sgf2n>::purge()
{
  DataFp.purge();
  DataF2.purge();
  DataFb.purge();
}

#endif
