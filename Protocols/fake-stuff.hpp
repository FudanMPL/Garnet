#ifndef PROTOCOLS_FAKE_STUFF_HPP_
#define PROTOCOLS_FAKE_STUFF_HPP_

#include "Protocols/fake-stuff.h"
#include "Processor/Data_Files.h"
#include "Tools/benchmarking.h"
#include "Math/Setup.h"
#include "GC/CcdSecret.h"
#include "FHE/tools.h"

#include "Protocols/ShamirInput.hpp"

#include <fstream>

template<class T> class Share;
template<class T> class SemiShare;
template<class T> class ShamirShare;
template<class T> class MaliciousShamirShare;
template<class T, int L> class FixedVec;
template<class T, class V> class Share_;
template<class T> class SpdzWiseShare;
template<class T, int L> class MamaMac;
template<class> class MaliciousRep3Share;
template<class> class DealerShare;

namespace GC
{
template<int S> class TinySecret;
template<class T> class TinierSecret;
template<class T> class MaliciousCcdSecret;
}

template<class T, class U, class V, class W>
void make_share(Share_<T, W>* Sa,const U& a,int N,const V& key,PRNG& G)
{
  T x;
  W mac, y;
  mac = a * key;
  Share_<T, W> S;
  S.set_share(a);
  S.set_mac(mac);

  for (int i=0; i<N-1; i++)
    { x.randomize(G);
      y.randomize(G);
      Sa[i].set_share(x);
      Sa[i].set_mac(y);
      S.sub(S,Sa[i]);
    }
  Sa[N-1]=S;
}

template<class T, class U, class V>
void make_share(SpdzWiseShare<MaliciousRep3Share<T>>* Sa,const U& a,int N,const V& key,PRNG& G)
{
  assert (key[0] == key[1]);
  auto mac = a * key[0];
  FixedVec<typename V::value_type, 3> shares, macs;
  shares.randomize_to_sum(a, G);
  macs.randomize_to_sum(mac, G);

  for (int i = 0; i < N; i++)
    {
      MaliciousRep3Share<T> share, mac;
      share[0] = shares[i];
      share[1] = shares[positive_modulo(i - 1, 3)];
      mac[0] = macs[i];
      mac[1] = macs[positive_modulo(i - 1, 3)];
      Sa[i].set_share(share);
      Sa[i].set_mac(mac);
    }
}

template<class T, class U, class V>
void make_share(SpdzWiseShare<MaliciousShamirShare<T>>* Sa, const U& a, int N,
    const V& key, PRNG& G)
{
  vector<MaliciousShamirShare<T>> shares(N), macs(N);
  make_share(shares.data(), a, N, {}, G);
  make_share(macs.data(), a * key, N, {}, G);
  for (int i = 0; i < N; i++)
    {
      Sa[i].set_share(shares[i]);
      Sa[i].set_mac(macs[i]);
    }
}

template<class T, class U, class V>
void make_vector_share(T* Sa,const U& a,int N,const V& key,PRNG& G)
{
  int length = Sa[0].default_length;
  for (int i = 0; i < N; i++)
    Sa[i].resize_regs(length);
  for (int j = 0; j < length; j++)
    {
      typename T::part_type shares[N];
      make_share(shares, typename T::part_type::clear(a.get_bit(j)), N, key, G);
      for (int i = 0; i < N; i++)
        Sa[i].get_reg(j) = shares[i];
    }
}

template<int S, class U, class V>
void make_share(GC::TinySecret<S>* Sa, const U& a, int N, const V& key, PRNG& G)
{
  make_vector_share(Sa, a, N, key, G);
}

template<class T, class U, class V>
void make_share(GC::TinierSecret<T>* Sa, const U& a, int N, const V& key, PRNG& G)
{
  make_vector_share(Sa, a, N, key, G);
}

template<class T, class U>
void make_share(SemiShare<T>* Sa,const T& a,int N,const U&,PRNG& G)
{
  T x, S = a;
  for (int i=0; i<N-1; i++)
    {
      x.randomize(G);
      Sa[i] = x;
      S -= x;
    }
  Sa[N-1]=S;
}

template<class T, class U, class V, int L>
void make_share(MamaMac<V, L>* Sa,const T& a,int N,const U&,PRNG& G)
{
  T x, S = a;
  for (int i=0; i<N-1; i++)
    {
      x.randomize(G);
      Sa[i] = x;
      S -= x;
    }
  Sa[N-1]=S;
}

template<class T, class U>
void make_share(DealerShare<T>* Sa, const T& a, int N, const U&, PRNG& G)
{
  make_share((SemiShare<T>*) Sa, a, N - 1, U(), G);
}

template<class T, class U, class V>
void make_share(FixedVec<T, 2>* Sa, const V& a, int N, const U& key, PRNG& G);

template<class T, class U>
inline void make_share(vector<T>& Sa,
    const typename T::open_type& a, int N, const U& key,
    PRNG& G)
{
  Sa.resize(N);
  make_share(Sa.data(), a, N, key, G);
}

template<class T, class U, class V>
void make_share(FixedVec<T, 2>* Sa, const V& a, int N, const U& key, PRNG& G)
{
  (void) key;
  assert(N == 3);
  FixedVec<T, 3> add_shares;
  // hack
  add_shares.randomize_to_sum(a, G);
  for (int i=0; i<N; i++)
    {
      FixedVec<T, 2> share;
      share[0] = add_shares[(i + 1) % 3];
      share[1] = add_shares[i];
      Sa[i] = share;
    }
}

template<class T, class U, class V>
void make_share(FixedVec<T, 3>* Sa, const V& a, int N, const U& key, PRNG& G)
{
  (void) key;
  assert(N == 4);
  FixedVec<T, 4> add_shares;
  add_shares.randomize_to_sum(a, G);
  for (int i=0; i<N; i++)
    {
      FixedVec<T, 3> share;
      share[0] = add_shares[(i + 0) % 4];
      share[1] = add_shares[(i + 1) % 4];
      share[2] = add_shares[(i + 2) % 4];
      Sa[i] = share;
    }
}

template<class T>
class VanderStore
{
public:
  static vector<vector<T>> vandermonde;
};

template<class T>
vector<vector<T>> VanderStore<T>::vandermonde;

template<class T, class V>
void make_share(ShamirShare<T>* Sa, const V& a, int N,
    const typename ShamirShare<T>::mac_type&, PRNG& G)
{
  auto& vandermonde = VanderStore<T>::vandermonde;
  if (vandermonde.empty())
      vandermonde = ShamirInput<ShamirShare<T>>::get_vandermonde(N / 2, N);
  vector<T> randomness(N / 2);
  for (auto& x : randomness)
      x.randomize(G);
  for (int i = 0; i < N; i++)
  {
      auto& share = Sa[i];
      share = a;
      for (int j = 0; j < ShamirOptions::singleton.threshold; j++)
          share += vandermonde.at(i).at(j) * randomness[j];
  }
}

template<class T, class V>
void check_share(vector<Share<T> >& Sa,
  V& value,
  T& mac,
  int N,
  const T& key)
{
  value = (0);
  mac = (0);

  for (int i=0; i<N; i++)
    {
      value += (Sa[i].get_share());
      mac += (Sa[i].get_mac());
    }

  V res;
  res = value * key;
  if (res != mac)
    {
      cout << "Value:      " << value << endl;
      cout << "Input MAC:  " << mac << endl;
      cout << "Actual MAC: " << res << endl;
      cout << "MAC key:    " << key << endl;
      throw mac_fail();
    }
}

template<class T>
void check_share(vector<T>& Sa, typename T::clear& value,
    typename T::mac_type& mac, int N, const typename T::mac_key_type& key)
{
  assert(N == 3);
  value = 0;
  (void)key;
  (void)mac;

  for (int i = 0; i < N; i++)
    {
      auto share = Sa[i];
      value += share[0];
      auto a = share[1];
      auto b = Sa[positive_modulo(i - 1, N)][0];
      if (a != b)
      {
        cout << a << " != " << b << endl;
        for (int i = 0; i < N; i++)
          cout << Sa[i] << endl;
        throw bad_value("invalid replicated secret sharing");
      }
    }
}

template<class T>
inline string mac_filename(string directory, int playerno)
{
  if (directory.empty())
    directory = ".";
  return directory + "/Player-MAC-Keys-" + T::type_short() + "-P"
      + to_string(playerno);
}

template <>
inline void write_mac_key(const string&, int, int, GC::NoValue)
{
}

template <class U>
void write_mac_key(const string& directory, int i, int nplayers, U key)
{
  ofstream outf;
  stringstream filename;
  filename << mac_filename<U>(directory, i);
#ifdef VERBOSE
  cout << "Writing to " << filename.str().c_str() << endl;
#endif
  outf.open(filename.str().c_str());
  outf << nplayers << endl;
  key.output(outf,true);
  outf.close();
}

template <class T>
void write_mac_key(const Names& N, typename T::mac_key_type key)
{
  write_mac_key(get_prep_sub_dir<T>(N.num_players()), N.my_num(),
      N.num_players(), key);
}

template <class T>
void read_mac_key(const string& directory, const Names& N, T& key)
{
  read_mac_key(directory, N.my_num(), N.num_players(), key);
}

template <>
inline void read_mac_key(const string&, int, int, GC::NoValue&)
{
}

template <class U>
void read_mac_key(const string& directory, int player_num, int nplayers, U& key)
{
  int nn;

  string filename = mac_filename<U>(directory, player_num);
  ifstream inpf;
#ifdef VERBOSE
  cerr << "Reading MAC keys from " << filename << endl;
#endif
  inpf.open(filename);
  if (inpf.fail())
    {
#ifdef VERBOSE
      cerr << "Could not open MAC key file. Perhaps it needs to be generated?\n";
#endif
      throw mac_key_error(filename);
    }
  inpf >> nn;
  if (nn!=nplayers)
    { cerr << "KeyGen was last run with " << nn << " players." << endl;
      cerr << "  - You are running Online with " << nplayers << " players." << endl;
      throw mac_key_error(filename);
    }

  try
  {
      key.input(inpf,true);
  }
  catch(exception&)
  {
      throw mac_key_error(filename);
  }

  if (inpf.fail())
      throw mac_key_error(filename);

  inpf.close();
}

template<class T>
typename T::mac_key_type read_generate_write_mac_key(Player& P,
        string directory)
{
  if (directory == "")
    directory = get_prep_sub_dir<T>(P.num_players());
  typename T::mac_key_type res, tmp;

  try
  {
      read_mac_key(directory, P.my_num(), P.num_players(), tmp);
  }
  catch (mac_key_error&)
  {
  }

  T::read_or_generate_mac_key(directory, P, res);

  // only write if changed
  if (tmp != res)
      write_mac_key(directory, P.my_num(), P.num_players(), res);

  return res;
}

template<>
inline GC::NoValue read_generate_write_mac_key<GC::NoShare>(Player&,
        string)
{
  return {};
}

template <class U>
void read_global_mac_key(const string& directory, int nparties, U& key)
{
  U pp;
  key.assign_zero();

  for (int i= 0; i < nparties; i++)
    {
      read_mac_key(directory, i, nparties, pp);
      cout << " Key " << i << ": " << pp << endl;
      key += pp;
    }

  cout << "--------------\n";
  cout << "Final Keys : " << key << endl;
}

template <>
inline void read_global_mac_key(const string&, int, GC::NoValue&)
{
}

template <class T>
T reconstruct(vector<T>& shares)
{
  return sum(shares);
}

template <class T>
T reconstruct(vector<MaliciousRep3Share<T>>& shares)
{
  T res;
  for (auto& x : shares)
    res += x[0];
  return res;
}

template <class T>
T reconstruct(vector<MaliciousShamirShare<T>>& shares)
{
  T res;
  for (size_t i = 0; i < shares.size(); i++)
    res += Shamir<ShamirShare<T>>::get_rec_factor(i, shares.size()) * shares[i];
  return res;
}

template<class T>
void make_mac_key_share(typename T::mac_share_type::open_type& key,
    vector<typename T::mac_share_type>& key_shares, int nplayers, T)
{
  SeededPRNG G;
  key.randomize(G);
  make_share(key_shares.data(), key, nplayers, GC::NoShare(), G);
  assert(not key_shares[0].is_zero());
}

template<int K, int S>
void make_mac_key_share(Z2<K + S>& key,
    vector<SemiShare<Z2<K + S>>>& key_shares, int nplayers, Spdz2kShare<K, S>)
{
  SeededPRNG G;
  key = {};
  key_shares.resize(nplayers);
  for (int i = 0; i < nplayers; i++)
    {
      key_shares[i] = G.get<Z2<S>>();
      key += key_shares[i];
    }
  assert(not key.is_zero());
}

template<class T>
void generate_mac_keys(typename T::mac_share_type::open_type& key,
    int nplayers, string prep_data_prefix)
{
  key.assign_zero();
  int tmpN = 0;
  ifstream inpf;
  prep_data_prefix = get_prep_sub_dir<T>(prep_data_prefix, nplayers);
  bool generate = false;
  vector<typename T::mac_share_type> key_shares(nplayers);

  for (int i = 0; i < nplayers; i++)
    {
      auto& pp = key_shares[i];
      stringstream filename;
      filename << mac_filename<typename T::mac_key_type>(prep_data_prefix, i);
      inpf.open(filename.str().c_str());
      if (inpf.fail())
        {
          inpf.close();
          cout << "No MAC key share for player " << i << ", generating a fresh ones\n";
          generate = true;
          break;
        }
      else
        {
          inpf >> tmpN; // not needed here
          pp.input(inpf,true);
          inpf.close();
          if (pp.is_zero())
            {
              generate = true;
              break;
            }
        }
      cout << " Key " << i << ": " << pp << endl;
    }

  key = reconstruct(key_shares);

  if (generate)
    {
      make_mac_key_share(key, key_shares, nplayers, T());

      for (int i = 0; i < nplayers; i++)
        {
          auto& pp = key_shares[i];
          stringstream filename;
          filename
              << mac_filename<typename T::mac_key_type>(prep_data_prefix, i);
          ofstream outf(filename.str().c_str());
          if (outf.fail())
            throw file_error(filename.str().c_str());
          outf << nplayers << " " << pp << endl;
          outf.close();
          cout << "Written new MAC key share to " << filename.str() << endl;
          cout << " Key " << i << ": " << pp << endl;
        }
    }

  cout << "--------------\n";
  cout << "Final Key: " << key << endl;
}

inline void check_files(ofstream* outf, int N)
{
  for (int i = 0; i < N; i++)
    if (outf[i].fail())
      throw runtime_error("couldn't write to file");
}

/* N      = Number players
 * ntrip  = Number triples needed
 */
template<class T>
void make_mult_triples(const typename T::mac_type& key, int N, int ntrip,
    bool zero, string prep_data_prefix, int thread_num = -1)
{
  T::clear::write_setup(get_prep_sub_dir<T>(prep_data_prefix, N));

  PRNG G;
  G.ReSeed();

  Files<T> files(N, key, prep_data_prefix, DATA_TRIPLE, thread_num);
  typename T::clear a,b,c;
  /* Generate Triples */
  for (int i=0; i<ntrip; i++)
    {
      if (!zero)
        a.randomize(G);
      if (!zero)
        b.randomize(G);
      auto c = typename T::open_type(a) * b;
      files.output_shares(a);
      files.output_shares(b);
      files.output_shares(c);
    }
  check_files(files.outf, N);
}

/* N      = Number players
 * ntrip  = Number inverses needed
 */
template<class T>
void make_inverse(const typename T::mac_type& key, int N, int ntrip, bool zero,
    string prep_data_prefix)
{
  PRNG G;
  G.ReSeed();

  Files<T> files(N, key, prep_data_prefix, DATA_INVERSE);
  typename T::clear a,b;
  for (int i=0; i<ntrip; i++)
    {
      if (zero)
        // ironic?
        a.assign_one();
      else
        do
          a.randomize(G);
        while (a.is_zero());
      files.output_shares(a);
      files.output_shares(a.invert());
    }
  check_files(files.outf, N);
}

template<class T>
void plain_edabits(vector<typename T::clear>& as,
    vector<typename T::bit_type::part_type::clear>& bs, int length, PRNG& G,
    bool zero = false)
{
  int max_size = edabitvec<T>::MAX_SIZE;
  as.resize(max_size);
  bs.clear();
  bs.resize(length);
  Z2<T::clear::MAX_EDABITS> value;
  for (int j = 0; j < max_size; j++)
    {
      if (not zero)
        value.randomize_part(G, length);
      as[j] = value;
      for (int k = 0; k < length; k++)
        bs[k] ^= BitVec(value.get_bit(k)) << j;
    }

}

#endif
