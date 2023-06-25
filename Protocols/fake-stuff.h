
#ifndef _fake_stuff
#define _fake_stuff

#include <fstream>
using namespace std;

#include "Networking/Player.h"
#include "Processor/Data_Files.h"
#include "Math/Setup.h"
#include "Tools/benchmarking.h"

template<class T>
void check_share(vector<T>& Sa, typename T::clear& value,
    typename T::mac_type& mac, int N, const typename T::mac_key_type& key);

template<class T> class Share;

template<class T, class V>
void check_share(vector<Share<T> >& Sa,
  V& value,
  T& mac,
  int N,
  const T& key);

// Generate MAC key shares
void generate_keys(const string& directory, int nplayers);

template <class T>
void write_mac_key(const string& directory, int player_num, int nplayers, T key);

template <class U>
void read_mac_key(const string& directory, int player_num, int nplayers, U& key);
template <class U>
void read_mac_key(const string& directory, const Names& N, U& key);

template <class T>
typename T::mac_key_type read_generate_write_mac_key(Player& P,
        string directory = "");

template <class T>
class Files
{
public:
  ofstream* outf;
  int N;
  typename T::mac_type key;
  PRNG G;
  Files(int N, const typename T::mac_type& key, const string& prep_data_prefix,
      Dtype type, int thread_num = -1) :
      Files(N, key,
          get_prep_sub_dir<T>(prep_data_prefix, N)
              + DataPositions::dtype_names[type] + "-" + T::type_short(),
          thread_num)
  {
  }
  Files(int N, const typename T::mac_type& key, const string& prefix,
      int thread_num = -1) :
      N(N), key(key)
  {
    insecure_fake(false);
    outf = new ofstream[N];
    for (int i=0; i<N; i++)
      {
        stringstream filename;
        filename << prefix << "-P" << i;
        filename << PrepBase::get_suffix(thread_num);
        cout << "Opening " << filename.str() << endl;
        outf[i].open(filename.str().c_str(),ios::out | ios::binary);
        file_signature<T>().output(outf[i]);
        if (outf[i].fail())
          throw file_error(filename.str().c_str());
      }
    G.ReSeed();
  }
  ~Files()
  {
    delete[] outf;
  }
  template<class U = T>
  void output_shares(const typename U::open_type& a)
  {
    output_shares<T>(a, key);
  }
  template<class U>
  void output_shares(const typename U::open_type& a,
      const typename U::mac_type& key)
  {
    vector<U> Sa(N);
    make_share(Sa,a,N,key,G);
    for (int j=0; j<N; j++)
      Sa[j].output(outf[j],false);
  }
};

#endif
