#ifndef PROCESSOR_DATA_FILES_HPP_
#define PROCESSOR_DATA_FILES_HPP_

#include "Processor/Data_Files.h"
#include "Processor/Processor.h"
#include "Processor/NoFilePrep.h"
#include "Protocols/dabit.h"
#include "Math/Setup.h"
#include "GC/BitPrepFiles.h"
#include "Tools/benchmarking.h"

template<class T>
Preprocessing<T>* Preprocessing<T>::get_live_prep(SubProcessor<T>* proc,
    DataPositions& usage)
{
  return new typename T::LivePrep(proc, usage);
}

template<class T>
template<class U, class V>
Preprocessing<T>* Preprocessing<T>::get_new(
    Machine<U, V>& machine,
    DataPositions& usage, SubProcessor<T>* proc)
{
  if (machine.live_prep)
    return get_live_prep(proc, usage);
  else
    return new Sub_Data_Files<T>(machine.get_N(),
        machine.template prep_dir_prefix<T>(), usage, BaseMachine::thread_num);
}

template<class T>
template<int>
Preprocessing<T>* Preprocessing<T>::get_new(
    bool live_prep, const Names& N,
    DataPositions& usage)
{
  if (live_prep)
    return new typename T::LivePrep(usage);
  else
    return new GC::BitPrepFiles<T>(N,
        get_prep_sub_dir<T>(PREP_DIR, N.num_players()), usage,
        BaseMachine::thread_num);
}

template<class T>
T Preprocessing<T>::get_random_from_inputs(int nplayers)
{
  T res;
  for (int j = 0; j < nplayers; j++)
    {
      T tmp;
      typename T::open_type _;
      this->get_input_no_count(tmp, _, j);
      res += tmp;
    }
  return res;
}

template<class T>
Sub_Data_Files<T>::Sub_Data_Files(const Names& N, DataPositions& usage,
    int thread_num) :
    Sub_Data_Files(N,
        OnlineOptions::singleton.prep_dir_prefix<T>(N.num_players()), usage,
        thread_num)
{
}


template<class T>
int Sub_Data_Files<T>::tuple_length(int dtype)
{
  return DataPositions::tuple_size[dtype] * T::size();
}

template<class T>
string Sub_Data_Files<T>::get_filename(const Names& N, Dtype type,
    int thread_num)
{
  return PrepBase::get_filename(get_prep_sub_dir<T>(N.num_players()),
      type, T::type_short(), N.my_num(), thread_num);
}

template<class T>
string Sub_Data_Files<T>::get_input_filename(const Names& N, int input_player,
    int thread_num)
{
  return PrepBase::get_input_filename(
      get_prep_sub_dir<T>(N.num_players()), T::type_short(), input_player,
      N.my_num(), thread_num);
}

template<class T>
string Sub_Data_Files<T>::get_edabit_filename(const Names& N, int n_bits,
    int thread_num)
{
  return PrepBase::get_edabit_filename(
      get_prep_sub_dir<T>(N.num_players()), n_bits, N.my_num(), thread_num);
}

template<class T>
Sub_Data_Files<T>::Sub_Data_Files(int my_num, int num_players,
    const string& prep_data_dir, DataPositions& usage, int thread_num) :
    Preprocessing<T>(usage),
    my_num(my_num), num_players(num_players), prep_data_dir(prep_data_dir),
    thread_num(thread_num), part(0)
{
#ifdef DEBUG_FILES
  cerr << "Setting up Data_Files in: " << prep_data_dir << endl;
#endif
  T::clear::check_setup(prep_data_dir);
  string type_short = T::type_short();
  string type_string = T::type_string();

  for (int dtype = 0; dtype < N_DTYPE; dtype++)
    {
      if (T::clear::allows(Dtype(dtype)))
        {
          buffers[dtype].setup(
              PrepBase::get_filename(prep_data_dir, Dtype(dtype), type_short,
                  my_num, thread_num), tuple_length(dtype), type_string,
              DataPositions::dtype_names[dtype]);
        }
    }

  dabit_buffer.setup(
      PrepBase::get_filename(prep_data_dir, DATA_DABIT,
          type_short, my_num, thread_num), dabit<T>::size(), type_string,
      DataPositions::dtype_names[DATA_DABIT]);

  input_buffers.resize(num_players);
  for (int i=0; i<num_players; i++)
    {
      string filename = PrepBase::get_input_filename(prep_data_dir,
          type_short, i, my_num, thread_num);
      if (i == my_num)
        my_input_buffers.setup(filename,
            T::size() + T::clear::size(), type_string);
      else
        input_buffers[i].setup(filename,
            T::size(), type_string);
    }

#ifdef DEBUG_FILES
  cerr << "done\n";
#endif
}

template<class sint, class sgf2n>
Data_Files<sint, sgf2n>::Data_Files(Machine<sint, sgf2n>& machine, SubProcessor<sint>* procp,
    SubProcessor<sgf2n>* proc2) :
    usage(machine.get_N().num_players()),
    DataFp(*Preprocessing<sint>::get_new(machine, usage, procp)),
    DataF2(*Preprocessing<sgf2n>::get_new(machine, usage, proc2)),
    DataFb(
        *Preprocessing<typename sint::bit_type>::get_new(machine.live_prep,
            machine.get_N(), usage))
{
}

template<class sint, class sgf2n>
Data_Files<sint, sgf2n>::Data_Files(const Names& N) :
    usage(N.num_players()),
    DataFp(*new Sub_Data_Files<sint>(N, usage)),
    DataF2(*new Sub_Data_Files<sgf2n>(N, usage)),
    DataFb(*new Sub_Data_Files<typename sint::bit_type>(N, usage))
{
}


template<class sint, class sgf2n>
Data_Files<sint, sgf2n>::~Data_Files()
{
  delete &DataFp;
  delete &DataF2;
  delete &DataFb;
}

template<class T>
Sub_Data_Files<T>::~Sub_Data_Files()
{
  for (auto& x: edabit_buffers)
    {
      delete x.second;
    }
  if (part != 0)
    delete part;
}

template<class T>
void Sub_Data_Files<T>::seekg(DataPositions& pos)
{
  if (OnlineOptions::singleton.file_prep_per_thread)
    return;

  if (T::LivePrep::use_part)
    {
      get_part().seekg(pos);
      return;
    }

  DataFieldType field_type = T::clear::field_type();
  for (int dtype = 0; dtype < N_DTYPE; dtype++)
    if (T::clear::allows(Dtype(dtype)))
      buffers[dtype].seekg(pos.files[field_type][dtype]);
  for (int j = 0; j < num_players; j++)
    if (j == my_num)
      my_input_buffers.seekg(pos.inputs[j][field_type]);
    else
      input_buffers[j].seekg(pos.inputs[j][field_type]);
  for (map<DataTag, long long>::const_iterator it = pos.extended[field_type].begin();
      it != pos.extended[field_type].end(); it++)
    {
      setup_extended(it->first);
      extended[it->first].seekg(it->second);
    }
  dabit_buffer.seekg(pos.files[field_type][DATA_DABIT]);
}

template<class sint, class sgf2n>
void Data_Files<sint, sgf2n>::seekg(DataPositions& pos)
{
  DataFp.seekg(pos);
  DataF2.seekg(pos);
  DataFb.seekg(pos);
  usage = pos;
}

template<class sint, class sgf2n>
void Data_Files<sint, sgf2n>::skip(const DataPositions& pos)
{
  DataPositions new_pos = usage;
  new_pos.increase(pos);
  skipped.increase(pos);
  seekg(new_pos);
}

template<class T>
void Sub_Data_Files<T>::prune()
{
  for (auto& buffer : buffers)
    buffer.prune();
  my_input_buffers.prune();
  for (int j = 0; j < num_players; j++)
    input_buffers[j].prune();
  for (auto& it : extended)
    it.second.prune();
  dabit_buffer.prune();
  if (part != 0)
    part->prune();
}

template<class sint, class sgf2n>
void Data_Files<sint, sgf2n>::prune()
{
  DataFp.prune();
  DataF2.prune();
  DataFb.prune();
}

template<class T>
void Sub_Data_Files<T>::purge()
{
  for (auto& buffer : buffers)
    buffer.purge();
  my_input_buffers.purge();
  for (int j = 0; j < num_players; j++)
    input_buffers[j].purge();
  for (auto it : extended)
    it.second.purge();
  dabit_buffer.purge();
  if (part != 0)
    part->purge();
}

template<class T>
void Sub_Data_Files<T>::setup_extended(const DataTag& tag, int tuple_size)
{
  auto& buffer = extended[tag];
  int tuple_length = tuple_size * T::size();

  if (!buffer.is_up())
    {
      stringstream ss;
      ss << prep_data_dir << tag.get_string() << "-" << T::type_short() << "-P" << my_num;
      buffer.setup(ss.str(), tuple_length);
    }

  buffer.check_tuple_length(tuple_length);
}

template<class T>
void Sub_Data_Files<T>::get_no_count(vector<T>& S, DataTag tag, const vector<int>& regs, int vector_size)
{
  setup_extended(tag, regs.size());
  for (int j = 0; j < vector_size; j++)
    for (unsigned int i = 0; i < regs.size(); i++)
      extended[tag].input(S[regs[i] + j]);
}

template<class T>
void Sub_Data_Files<T>::get_dabit_no_count(T& a, typename T::bit_type& b)
{
  dabit<T> tmp;
  dabit_buffer.input(tmp);
  a = tmp.first;
  b = tmp.second;
}

template<class T>
template<int>
void Sub_Data_Files<T>::buffer_edabits_with_queues(bool strict, int n_bits,
        false_type)
{
  insecure("reading edaBits from files");
  if (edabit_buffers.find(n_bits) == edabit_buffers.end())
    {
      string filename = PrepBase::get_edabit_filename(prep_data_dir,
          n_bits, my_num, thread_num);
      ifstream* f = new ifstream(filename);
      if (f->fail())
        throw runtime_error("cannot open " + filename);
      check_file_signature<T>(*f, filename);
      edabit_buffers[n_bits] = f;
    }
  auto& buffer = *edabit_buffers[n_bits];
  if (buffer.peek() == EOF)
    buffer.seekg(file_signature<T>().get_length());
  edabitvec<T> eb;
  eb.input(n_bits, buffer);
  this->edabits[{strict, n_bits}].push_back(eb);
  if (buffer.fail())
    throw runtime_error("error reading edaBits");
}

template<class T>
typename Sub_Data_Files<T>::part_type& Sub_Data_Files<T>::get_part()
{
  if (part == 0)
    part = new part_type(my_num, num_players,
        get_prep_sub_dir<typename T::part_type>(num_players), this->usage,
        thread_num);
  return *part;
}

#endif
