/*
 * Check-Offline.cpp
 *
 */

#include "Math/gf2n.h"
#include "Math/gfp.h"
#include "Protocols/Share.h"
#include "Protocols/fake-stuff.h"
#include "Protocols/MAC_Check.h"
#include "Protocols/Rep3Share.h"
#include "Tools/ezOptionParser.h"
#include "Tools/Exceptions.h"
#include "GC/MaliciousRepSecret.h"
#include "GC/TinierSecret.h"
#include "GC/TinyMC.h"
#include "GC/SemiSecret.h"
#include "GC/RepPrep.h"

#include "Math/Setup.h"
#include "Processor/Data_Files.h"

#include "Protocols/fake-stuff.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "Processor/Data_Files.hpp"
#include "Math/Z2k.hpp"
#include "Math/gfp.hpp"
#include "GC/Secret.hpp"

#include <sstream>
#include <fstream>
#include <vector>
using namespace std;

string PREP_DATA_PREFIX;

template<class T>
void check_mult_triples(const typename T::mac_key_type& key,int N,vector<Sub_Data_Files<T>*>& dataF)
{
  typename T::clear a,b,c;
  typename T::mac_type mac;
  vector<T> Sa(N),Sb(N),Sc(N);
  int n = 0;

  try {
      while (!dataF[0]->eof(DATA_TRIPLE))
        {
          for (int i = 0; i < N; i++)
            dataF[i]->get_three(DATA_TRIPLE, Sa[i], Sb[i], Sc[i]);
          check_share(Sa, a, mac, N, key);
          check_share(Sb, b, mac, N, key);
          check_share(Sc, c, mac, N, key);

          if (a * b != c)
            {
              cout << n << ": " << c << " != " << a << " * " << b << endl;
              throw bad_value();
            }
          n++;
        }

      cout << n << " triples of type " << T::type_string() << endl;
  }
  catch (exception& e)
  {
      cout << "Error '" << e.what() << "' after " << n << " triples of type " << T::type_string() << endl;
  }
}

template <class T>
void check_square(const T& a, const T& b, int n)
{
  if (a * a != b)
    {
      cout << n << ": " << b << " != " << a << "^2" << endl;
      throw bad_value();
    }
}

template <class T>
void check_inverse(const T& a, const T& b, int n)
{
  if (!(a * b).is_one())
    {
      cout << n << ": " << b << " != " << a << "^-1" << endl;
      throw bad_value();
    }
}

template<class T>
void check_tuple(const T& a, const T& b, int n, Dtype type)
{
  if (type == DATA_SQUARE)
    check_square(a, b, n);
  else if (type == DATA_INVERSE)
    check_inverse(a, b, n);
  else
    throw runtime_error("type not supported");
}

template<class T>
void check_tuples(const typename T::mac_key_type& key,int N,vector<Sub_Data_Files<T>*>& dataF, Dtype type)
{
  typename T::clear a,b,c,res;
  typename T::mac_type mac;
  vector<T> Sa(N),Sb(N),Sc(N);
  int n = 0;

  try {
      while (!dataF[0]->eof(type))
        {
          for (int i = 0; i < N; i++)
            dataF[i]->get_two(type, Sa[i], Sb[i]);
          check_share(Sa, a, mac, N, key);
          check_share(Sb, b, mac, N, key);
          check_tuple(a, b, n, type);
          n++;
        }

        cout << n << " " << DataPositions::dtype_names[type] << " of type "
                << T::type_string() << endl;
  }
  catch (exception& e)
  {
      cout << "Error after " << n << " " << DataPositions::dtype_names[type] <<
              " of type " << T::type_string() << endl;
  }
}

template<class T>
void check_bits(const typename T::mac_key_type& key,int N,vector<Sub_Data_Files<T>*>& dataF)
{
  typename T::clear a,b,c,res;
  typename T::mac_type mac;
  vector<T> Sa(N),Sb(N),Sc(N);
  int n = 0;

  try {
      while (!dataF[0]->eof(DATA_BIT))
      {
          for (int i = 0; i < N; i++)
              dataF[i]->get_one(DATA_BIT, Sa[i]);
          check_share(Sa, a, mac, N, key);

          if (!(a.is_zero() || a.is_one()))
          {
              cout << n << ": " << a << " neither 0 or 1" << endl;
              throw bad_value();
          }
          n++;
      }

      cout << n << " bits of type " << T::type_string() << endl;
  }
  catch (exception& e)
  {
      cout << "Error after " << n << " bits of type " << T::type_string() << endl;
  }
}

template<class T>
void check_inputs(const typename T::mac_key_type& key,int N,vector<Sub_Data_Files<T>*>& dataF)
{
  typename T::clear a, x;
  typename T::mac_type mac;
  vector<T> Sa(N);

  for (int player = 0; player < N; player++)
    {
      int n = 0;
      try {
          while (!dataF[0]->input_eof(player))
          {
              for (int i = 0; i < N; i++)
                  dataF[i]->get_input(Sa[i], x, player);
              check_share(Sa, a, mac, N, key);
              if (a != (x))
                  throw bad_value();
              n++;
          }
          cout << n << " input masks for player " << player << " of type " << T::type_string() << endl;
      }
      catch (exception& e)
      {
          cout << "Error after " << n << " input masks of type "
              << T::type_string() << " for player " << player << endl;
      }
  }
}

template<class T>
vector<Sub_Data_Files<T>*> setup(int N, DataPositions& usage, int thread_num = -1)
{
  vector<Sub_Data_Files<T>*> dataF(N);
  for (int i = 0; i < N; i++)
    dataF[i] = new Sub_Data_Files<T>(i, N,
        get_prep_sub_dir<T>(PREP_DATA_PREFIX, N), usage, thread_num);
  return dataF;
}

template<class T>
void check(int N, bool only_bits = false)
{
  typename T::mac_key_type key;
  read_global_mac_key(get_prep_sub_dir<T>(PREP_DATA_PREFIX, N), N, key);
  DataPositions usage(N);
  auto dataF = setup<T>(N, usage);
  check_bits(key, N, dataF);

  if (not only_bits)
  {
    check_mult_triples(key, N, dataF);
    check_inputs<T>(key, N, dataF);
    check_tuples<T>(key, N, dataF, DATA_SQUARE);
    check_tuples<T>(key, N, dataF, DATA_INVERSE);
  }
}

int main(int argc, const char** argv)
{
  ez::ezOptionParser opt;

  opt.syntax = "./Check-Offline.x <nparties> [OPTIONS]\n";
  opt.example = "./Check-Offline.x 3 -lgp 64 -lg2 128\n";

  opt.add(
        "128", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "Bit length of GF(p) field (default: 128)", // Help description.
        "-lgp", // Flag token.
        "--lgp" // Flag token.
  );
  opt.add(
        to_string(gf2n::default_degree()).c_str(), // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        ("Bit length of GF(2^n) field (default: " + to_string(gf2n::default_degree()) + ")").c_str(), // Help description.
        "-lg2", // Flag token.
        "--lg2" // Flag token.
  );
  opt.add(
        "", // Default.
        0, // Required?
        0, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "Read GF(p) triples in Montgomery representation (default: not set)", // Help description.
        "-m", // Flag token.
        "--usemont" // Flag token.
  );
  opt.add(
        "", // Default.
        0, // Required?
        1, // Number of args expected.
        0, // Delimiter if expecting multiple args.
        "Directory containing the data (default: " PREP_DIR "<nparties>-<lgp>-<lg2>", // Help description.
        "-d", // Flag token.
        "--dir" // Flag token.
  );
  opt.parse(argc, argv);

  string usage;
  int lgp, lg2, nparties;
  bool use_montgomery = false;
  opt.get("--lgp")->getInt(lgp);
  opt.get("--lg2")->getInt(lg2);
  if (opt.isSet("--usemont"))
        use_montgomery = true;

  if (opt.firstArgs.size() == 2)
      nparties = atoi(opt.firstArgs[1]->c_str());
  else if (opt.lastArgs.size() == 1)
      nparties = atoi(opt.lastArgs[0]->c_str());
  else
  {
      cerr << "ERROR: invalid number of arguments\n";
      opt.getUsage(usage);
      cout << usage;
      return 1;
  }

  if (opt.isSet("--dir"))
    {
      opt.get("--dir")->getString(PREP_DATA_PREFIX);
      PREP_DATA_PREFIX += "/";
    }
  else
    PREP_DATA_PREFIX = PREP_DIR;

  string prep_dir = get_prep_sub_dir<Share<gfp>>(PREP_DATA_PREFIX, nparties, lgp);

  try
    {
      read_setup(prep_dir);
    }
  catch (exception& e)
    {
      cerr << "Ignoring: " << e.what() << endl;
    }

  if (!use_montgomery)
  {
    // no montgomery
    gfp::init_field(gfp::pr(), false);
  }

  int N = nparties;

  try
    {
      check<Share<gfp>>(N);
    }
  catch (exception& e)
    {
      cerr << "Ignoring: " << e.what() << endl;
    }

  if (lg2 > 64)
    {
      gf2n_long::init_field(lg2);
      check<Share<gf2n_long>>(N);
    }
  else
    {
      gf2n_short::init_field(lg2);
      check<Share<gf2n_short>>(N);
    }

  if (N == 3)
    {
      DataPositions pos(N);
      auto dataF = setup<Rep3Share<Integer>>(N, pos);
      check_bits({}, N, dataF);

      check<Rep3Share<gfp>>({}, N);

      auto dataF2 = setup<GC::MaliciousRepSecret>(N, pos, 0);
      check_mult_triples({}, N, dataF2);
      check_bits({}, N, dataF2);
    }
}
