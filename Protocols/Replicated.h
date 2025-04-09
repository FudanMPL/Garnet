/*
 * Replicated.h
 *
 */

#ifndef PROTOCOLS_REPLICATED_H_
#define PROTOCOLS_REPLICATED_H_

#include <algorithm>
#include <array>
#include <assert.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;

#include "Networking/Player.h"
#include "OT/OTExtension.h"
#include "OT/OTExtensionWithMatrix.h"
#include "Tools/PointerVector.h"
#include "Tools/octetStream.h"
#include "Tools/random.h"
#include "Tools_PSI/OKVS.h"
#include "Tools_PSI/SimpleIndex.h"
#include "cryptoTools/Common/BitVector.h"
#include "cryptoTools/Common/CuckooIndex.h"
#include <typeinfo>

template <class T> class SubProcessor;
template <class T> class ReplicatedMC;
template <class T> class ReplicatedInput;
template <class T> class Preprocessing;
template <class T> class SecureShuffle;
template <class T> class Rep3Shuffler;
class Instruction;

#define RECEIVER_P 0
typedef uint64_t idtype;

/**
 * Base class for replicated three-party protocols
 */
class ReplicatedBase {
public:
  array<PRNG, 2> shared_prngs;

  Player &P;

  ReplicatedBase(Player &P);
  ReplicatedBase(Player &P, array<PRNG, 2> &prngs);

  ReplicatedBase branch();

  int get_n_relevant_players() { return P.num_players() - 1; }
};

/**
 * Abstract base class for multiplication protocols
 */
template <class T> class ProtocolBase {
  virtual void buffer_random() { throw not_implemented(); }

protected:
  vector<T> random;

  int trunc_pr_counter;
  int rounds, trunc_rounds;
  int dot_counter;
  int bit_counter;

public:
  typedef T share_type;

  typedef SecureShuffle<T> Shuffler;

  int counter;

  ProtocolBase();
  virtual ~ProtocolBase();

  void muls(const vector<int> &reg, SubProcessor<T> &proc,
            typename T::MAC_Check &MC, int size);
  void mulrs(const vector<int> &reg, SubProcessor<T> &proc);

  void multiply(vector<T> &products, vector<pair<T, T>> &multiplicands,
                int begin, int end, SubProcessor<T> &proc);

  /// Single multiplication
  T mul(const T &x, const T &y);

  /// Initialize protocol if needed (repeated call possible)
  virtual void init(Preprocessing<T> &, typename T::MAC_Check &) {}

  /// Initialize multiplication round
  virtual void init_mul() = 0;
  /// Schedule multiplication of operand pair
  virtual void prepare_mul(const T &x, const T &y, int n = -1) = 0;
  virtual void prepare_mult(const T &x, const T &y, int n, bool repeat);
  /// Run multiplication protocol
  virtual void exchange() = 0;
  /// Get next multiplication result
  virtual T finalize_mul(int n = -1) = 0;
  /// Store next multiplication result in ``res``
  virtual void finalize_mult(T &res, int n = -1);

  /// Initialize dot product round
  void init_dotprod() { init_mul(); }
  /// Add operand pair to current dot product
  void prepare_dotprod(const T &x, const T &y) { prepare_mul(x, y); }
  /// Finish dot product
  void next_dotprod() {}
  /// Get next dot product result
  T finalize_dotprod(int length);

  virtual T get_random();

  virtual void trunc_pr(const vector<int> &regs, int size,
                        SubProcessor<T> &proc) {
    (void)regs, (void)size;
    (void)proc;
    throw runtime_error("trunc_pr not implemented");
  }

  virtual void randoms(T &, int) {
    throw runtime_error("randoms not implemented");
  }
  virtual void randoms_inst(vector<T> &, const Instruction &);

  template <int = 0>
  void matmulsm(SubProcessor<T> &proc, CheckVector<T> &source,
                const Instruction &instruction) {
    proc.matmulsm(source, instruction);
  }

  template <int = 0>
  void conv2ds(SubProcessor<T> &proc, const Instruction &instruction) {
    proc.conv2ds(instruction);
  }

  virtual void start_exchange() { exchange(); }
  virtual void stop_exchange() {}

  virtual void check() {}

  virtual void cisc(SubProcessor<T> &, const Instruction &) {
    throw runtime_error("CISC instuctions not implemented");
  }
  virtual vector<int> get_relevant_players();
};

/**
 * Semi-honest replicated three-party protocol
 */
template <class T>
class Replicated : public ReplicatedBase, public ProtocolBase<T> {
  array<octetStream, 2> os;
  PointerVector<typename T::clear> add_shares;
  typename T::clear dotprod_share;

  template <class U>
  void trunc_pr(const vector<int> &regs, int size, U &proc, true_type);
  template <class U>
  void trunc_pr(const vector<int> &regs, int size, U &proc, false_type);

public:
  static const bool uses_triples = false;

  typedef Rep3Shuffler<T> Shuffler;

  Replicated(Player &P);
  Replicated(const ReplicatedBase &other);

  static void assign(T &share, const typename T::clear &value, int my_num) {
    assert(T::vector_length == 2);
    share.assign_zero();
    if (my_num < 2)
      share[my_num] = value;
  }

  void init_mul();
  void prepare_mul(const T &x, const T &y, int n = -1);
  void exchange();
  T finalize_mul(int n = -1);

  void prepare_reshare(const typename T::clear &share, int n = -1);

  void init_dotprod();
  void prepare_dotprod(const T &x, const T &y);
  void next_dotprod();
  T finalize_dotprod(int length);

  template <class U> void trunc_pr(const vector<int> &regs, int size, U &proc);

  T get_random();
  void randoms(T &res, int n_bits);

  void start_exchange();
  void stop_exchange();

  template <class U>
  void psi(const vector<typename T::clear> &source,
           const Instruction &instruction, U &proc) {
    throw not_implemented();
  }

  template <class U>
  void psi_align(const vector<typename T::clear> &source,
                 const Instruction &instruction, U &proc) {
    throw not_implemented();
  }

  template <class U> void mpsi(int64_t n, U &proc) {
    int my_num = proc.P.my_num();
#ifdef DEBUG
    std::cout << "begin mpsi in rss machine " << n << " num:" << my_num
              << std::endl;
#endif

    fstream r;
    uint64_t m = 0;
    r.open("Player-Data/PSI/ID-P" + to_string(proc.P.my_num()), ios::in);
    vector<osuCrypto::block> ids;
    vector<idtype> smallids;
    idtype r_tmp;
    while (r >> r_tmp) {
      smallids.push_back(r_tmp);
      ids.push_back(osuCrypto::block(r_tmp));
      m++;
    }
    r.close();
#ifdef DEBUG
    cout << "Total lines read: " << m << endl;
#endif
    // size_t mm = 1ull << static_cast<size_t>(std::ceil(std::log2(m + 1)));
#ifdef DEBUG
    cout << "Bin size: " << mm << endl;
#endif

    int ssp = 40;
    osuCrypto::CuckooParam params =
        oc::CuckooIndex<>::selectParams(m, ssp, 0, 3);
#ifdef DEBUG
    cout << "hash bins: " << params.numBins() << endl;
#endif
    int nbase = 128;
    size_t l = sizeof(idtype) * 8;
    int nOTs = l * params.numBins();
    // PRNG G;
    // G.ReSeed();
    OT_ROLE ot_role;
    osuCrypto::CuckooIndex<> cuckoo;
    SimpleIndex sIdx;
    uint expand = 10;
    uint64_t mm;

    // std::vector<string> names;
    // int names_size = n-1;
    // int portnum_base = 5000;
    // Names *N = new Names(my_num, portnum_base + 1000 * names_size, names);

    // std::cout << proc.P.N.
    if (proc.P.my_num() == RECEIVER_P) {
      int seed = 1024;
      osuCrypto::block cuckooSeed(seed);
      cuckoo.init(params);
      cuckoo.insert(ids, cuckooSeed);
      // cuckoo.print();
      ot_role = RECEIVER;

      int i = 1;
      octetStream cs0;
      cs0.store(seed);
      for (int j = 1; j < n; j++) {
        proc.P.send_to(j, cs0);
      }
      // std::cout << "receive maxbins" << std::endl;
      octetStream cs_b;
      proc.P.receive_player(my_num + 1, cs_b);
      uint64_t maxBins;
      cs_b.get(maxBins);
      std::cout << "update maxBin: " << maxBins << std::endl;
      mm = maxBins * expand;
    } else if (my_num == 1) {
      octetStream cs0, cs1, cs_v, cs_s;
      proc.P.receive_player(RECEIVER_P, cs0);
      int seed;
      cs0.get(seed);
      osuCrypto::block cuckooSeed(seed);
      sIdx.init(params.numBins(), m, ssp, 3);
      sIdx.insertItems(ids, cuckooSeed);
      uint64_t maxBins = 0;
      for (unsigned int i = 0; i < params.numBins(); i++) {
        maxBins = max(maxBins, sIdx.mBinSizes[i]);
      }
      std::cout << "maxBins: " << maxBins << std::endl;

      octetStream cs_bs[n - 2];
      for (unsigned int i = 0; i < n - 2; i++) {
        proc.P.receive_player(i + my_num + 1, cs_bs[i]);
        uint64_t b;
        cs_bs[i].get(b);
        // std::cout << i << " " << b << std::endl;
        maxBins = max(maxBins, b);
      }
      // octetStream cs_mb[n];
      for (unsigned int i = 0; i < n; i++) {
        if (i != my_num) {
          // std::cout << "send to " << i << std::endl;
          octetStream cs_mb;
          cs_mb.store(maxBins);
          proc.P.send_to(i, cs_mb);
        }
      }
      std::cout << "update maxBin: " << maxBins << std::endl;
      mm = maxBins * expand;
    } else {
      octetStream cs0, cs_b;
      proc.P.receive_player(RECEIVER_P, cs0);
      int seed;
      cs0.get(seed);
#ifdef DEBUG
      std::cout << "Receive seed " << seed << std::endl;
#endif
      osuCrypto::block cuckooSeed(seed);
      sIdx.init(params.numBins(), m, ssp, 3);
      sIdx.insertItems(ids, cuckooSeed);
      uint64_t maxBins = 0;
      for (unsigned int i = 0; i < params.numBins(); i++) {
        maxBins = max(maxBins, sIdx.mBinSizes[i]);
        // if (sIdx.mBinSizes[i] > mm) {
        //   cout << "exceed" << endl;
        // }
      }
      std::cout << "maxBins: " << maxBins << std::endl;
      cs_b.store(maxBins);
      proc.P.send_to(RECEIVER_P + 1, cs_b);
      octetStream cs_mb;
      proc.P.receive_player(RECEIVER_P + 1, cs_mb);
      cs_mb.get(maxBins);
      std::cout << "update maxBin: " << maxBins << std::endl;
      mm = maxBins * expand;
    }

    if (proc.P.my_num() == RECEIVER_P) {
#ifdef DEBUG
      cout << "1.OPRF with:" << i << "-----------" << std::endl;
      cout << "(1) begin base ot \n";
#endif
      // perform OTs with player 1
      // string id_name = "Machine" + to_string(i);
      RealTwoPartyPlayer *rP =
          new RealTwoPartyPlayer(proc.P.N, RECEIVER_P + 1, "machine");
      BaseOT bot = BaseOT(nbase, 128, rP, INV_ROLE(ot_role));
      bot.exec_base();

      octetStream cs1;
      int a = 3;
      cs1.store(a);
      proc.P.send_to(RECEIVER_P + 1, cs1);
      cout << "send sync" << a << endl;

      // convert baseOT selection bits to BitVector
      // (not already BitVector due to legacy PVW code)
      BitVector receiverInput;
      BitVector baseReceiverInput = bot.receiver_inputs;
      baseReceiverInput.resize(nbase);
      OTExtensionWithMatrix *ot_ext = new OTExtensionWithMatrix(rP, ot_role);
      receiverInput = BitVector(nOTs);

      idtype idx;
      for (size_t i = 0; i < params.numBins(); i++) {
        if (!cuckoo.mBins[i].isEmpty()) {
          idx = smallids[cuckoo.mBins[i].idx()];
          // cout << idx << " | ";
          receiverInput.set_word(i, idx);
          // cout << receiverInput.get_word(i) << endl;
        }
      }

      ot_ext->init(baseReceiverInput, bot.sender_inputs, bot.receiver_outputs);
      ot_ext->transfer(nOTs, receiverInput, 1);
      // ot_ext.check();
      bot.extend_length();
#ifdef DEBUG
      cout << "(3) caculate oprf" << a << endl;
#endif
      BitVector key, temp;
      // string strkey;
      key.resize(sizeof(__m128i) << 3);
      std::vector<BitVector> fk_id(params.numBins());
      for (unsigned int i = 0; i < params.numBins(); i++) {
        if (!cuckoo.mBins[i].isEmpty()) {
          // cout << i << ": ";
          key.assign_zero();
          // get oprf
          for (unsigned int j = 0; j < l; j++) {
            temp.assign_bytes((char *)ot_ext->get_receiver_output(i * l + j),
                              sizeof(__m128i));
            // cout << temp.str() << endl;
            key.add(temp);
          }
          // r_fs[i] = key;
          // find same element
          // strkey = key.str();
          idtype inter_id = smallids[cuckoo.mBins[i].idx()];
#ifdef DEBUG
          cout << inter_id << "|" << key.str() << endl;
#endif
          fk_id[i] = key;
        }
      }

      delete ot_ext;
      rP->~RealTwoPartyPlayer();
      // #ifdef DEBUG
      cout << "2.receive okvs from:" << n - 1 << "-----------" << std::endl;
      // #endif
      octetStream cs_v, cs_s;
      proc.P.receive_player(n - 1, cs_v);
      vector<uint64_t> vs;
      uint64_t v;
      for (unsigned int i = 0; i < params.numBins() * 3; i++) {
        cs_v.get(v);
        vs.push_back(v);
      }

      proc.P.receive_player(n - 1, cs_s);
      std::pair<uint64_t, BitVector> idx_tmp;
      vector<BitVector> S;
      vector<idtype> I; // intersection IDs
      idtype id;
      uint64_t v1, v2, v3;
      for (unsigned int i = 0; i < params.numBins(); i++) {
        v1 = vs[3 * i];
        v2 = vs[3 * i + 1];
        v3 = vs[3 * i + 2];
#ifdef DEBUG
        cout << "Bin #" << i << endl;
#endif
        S.clear();
        // S.resize(mm);
        BitVector s_tmp;
        for (unsigned int j = 0; j < mm; j++) {
          s_tmp.unpack(cs_s);
          S.push_back(s_tmp);
        }

        OKVSReceiver okvs_r(S, v1, v2, v3, mm);
        if (!cuckoo.mBins[i].isEmpty()) {
          id = smallids[cuckoo.mBins[i].idx()];
          // std::cout << i << " " << id << " " << fk_id[i].str() << std::endl;
          idx_tmp = okvs_r.get(id);
// std::cout << "get from okvs:" << std::endl;
#ifdef DEBUG
          std::cout << id << "," << idx_tmp.first << "| "
                    << idx_tmp.second.str() << std::endl;
#endif
          if (idx_tmp.second.str() == fk_id[i].str()) {
            I.push_back(id);
            // std::cout << i << " " << id << endl;
          }
        }
      }
      const std::string filename = "Player-Data/PSI/ID-InterSet";
      std::ofstream outFile(filename);
      if (!outFile.is_open()) {
        std::cerr << "Cannot open: " << filename << std::endl;
      } else {
        for (const auto &id : I) {
          outFile << id << "\n";
        }
        outFile.close();
      }

    } else if (my_num == 1) {
      ot_role = SENDER;
#ifdef DEBUG
      cout << "Begin base ot \n";
#endif
      // string id_name = "Machine" + to_string(proc.P.my_num());
      RealTwoPartyPlayer *rP =
          new RealTwoPartyPlayer(proc.P.N, RECEIVER_P, "machine");
      BaseOT bot = BaseOT(nbase, 128, rP, INV_ROLE(ot_role));
      bot.exec_base();

      octetStream cs1;
      proc.P.receive_player(RECEIVER_P, cs1);
      int a;
      cs1.get(a);
      cout << "Receive sync" << a << endl;

      BitVector baseReceiverInput = bot.receiver_inputs;
      baseReceiverInput.resize(nbase);

      OTExtensionWithMatrix *ot_ext = new OTExtensionWithMatrix(rP, ot_role);
      BitVector receiverInput(nOTs);

      ot_ext->init(baseReceiverInput, bot.sender_inputs, bot.receiver_outputs);
      ot_ext->transfer(nOTs, receiverInput, 1);
      // ot_ext.check();
      bot.extend_length();

      // #ifdef DEBUG
      cout << "finish oprf with receiver\n";
      // #endif

      BitVector key, temp;
      idtype id;
      // vector<vector<BitVector>> fkx(params.numBins());
      key.resize(sizeof(__m128i) << 3);
      vector<BitVector> Y;
      vector<uint64_t> X;
      std::unordered_set<uint64_t> X_tmp;
      octetStream cs_s, cs_v;
      for (unsigned int i = 0; i < params.numBins(); i++) {
#ifdef DEBUG
        cout << "Bin #" << i << endl;
#endif
        X.clear();
        Y.clear();
        X_tmp.clear();
        std::array<vector<BitVector>, 2> outs;
        ot_ext->get_sender_output128i(outs, i * l, l);

        for (unsigned int k = 0; k < sIdx.mBinSizes[i]; k++) { // each bin
          key.assign_zero();
          id = smallids[sIdx.mBins(i, k).idx()];
#ifdef DEBUG
          std::cout << id << "" << std::endl;
#endif
          if (!X_tmp.empty() && X_tmp.find(id) != X_tmp.end())
            break;
          X.push_back(id);
          X_tmp.insert(id);
#ifdef DEBUG
          cout << id << " | ";
#endif
          for (unsigned int j = 0; j < l; j++) {
            // cout << (id & 0x1);
            temp = outs[id & 0x1][j];
            id = id >> 1;
            key.add(temp);
          }
// fs.push_back(key);
#ifdef DEBUG
          cout << key.str() << endl;
#endif
          Y.push_back(key);
          // cout << "|   " << fkx[i][k].str() << endl;
        }
        std::vector<BitVector> S;
        uint64_t v1 = 0;
        uint64_t v2 = 0;
        uint64_t v3 = 0;
        OKVSSender::generateTable(X, Y, mm, S, v1, v2, v3);
        cs_v.store(v1);
        cs_v.store(v2);
        cs_v.store(v3);
        // if (i % 10000 == 0) {
        //   cout << "finish build okvs " << i << endl;
        // }
// cs_s.store<BitVector>(s);
#ifdef DEBUG
        std::cout << "table, v: " << v1 << " " << v2 << " " << v3 << std::endl;
#endif
        for (auto &s : S) {
          s.pack(cs_s);
#ifdef DEBUG
          std::cout << s.str() << std::endl;
#endif
        }
      }
      std::cout << "begin send okvs\n";
      proc.P.send_to(my_num + 1, cs_v);
      proc.P.send_to(my_num + 1, cs_s);
      // for (BitVector fk : fs) {
      //   std::cout << fk.str() << std::endl;
      //   // fk.pack(cs2);
      // }
      std::cout << "finish send okvs\n";
      delete ot_ext;
      // delete rP;

    } else {
      octetStream cs_v, cs_s;
      proc.P.receive_player(my_num - 1, cs_v);
      octetStream cs_snew;
      uint64_t v;
      uint64_t v1, v2, v3;
      idtype id;
      std::pair<uint64_t, BitVector> idx_tmp;
      vector<BitVector> S;
      vector<BitVector> newS;
      vector<uint64_t> vs;
      for (unsigned int i = 0; i < params.numBins() * 3; i++) {
        cs_v.get(v);
        vs.push_back(v);
      }

      proc.P.receive_player(my_num - 1, cs_s);

      // update S to newS
      // PRNG seed;
      // seed.ReSeed();
      PRNG prng;
      prng.InitSeed();
      uint64_t rand_x[2];
      BitVector x_bitv;
      for (unsigned int i = 0; i < params.numBins(); i++) {
        v1 = vs[i * 3];
        v2 = vs[i * 3 + 1];
        v3 = vs[i * 3 + 2];
#ifdef DEBUG
        cout << "Bin #" << i << ",  v: " << v1 << " " << v2 << " " << v3
             << endl;
#endif
        S.clear();
        // S.resize(mm);
        BitVector s_tmp;
        for (unsigned int j = 0; j < mm; j++) {
          // std::cout << j << " ";
          s_tmp.unpack(cs_s);
          // std::cout << s_tmp.str() << std::endl;
          S.push_back(s_tmp);
        }

        OKVSReceiver okvs_r(S, v1, v2, v3, mm);

        newS.clear();
        // newS.resize(mm);
        for (size_t j = 0; j < mm; j++) {
          rand_x[0] = prng.get<uint64_t>();
          rand_x[1] = prng.get<uint64_t>();
          // x_128 = prng.get<__m128i>();
          x_bitv = BitVector(reinterpret_cast<uint8_t *>(rand_x), 16);
          newS.push_back(x_bitv);
          // x_bitv
        }
#ifdef DEBUG
        std::cout << "id, idx, s" << std::endl;
#endif
        for (unsigned int k = 0; k < sIdx.mBinSizes[i]; k++) { // each bin
          id = smallids[sIdx.mBins(i, k).idx()];
          idx_tmp = okvs_r.get(id);
#ifdef DEBUG
          std::cout << id << "," << idx_tmp.first << "|" << idx_tmp.second.str()
                    << std::endl;
#endif
          newS[idx_tmp.first] = idx_tmp.second;
        }
#ifdef DEBUG
        std::cout << "Table:\n";
#endif
        for (auto &s : newS) {
          s.pack(cs_snew);
#ifdef DEBUG
          std::cout << s.str() << std::endl;
#endif
        }
      }
      proc.P.send_to((my_num + 1) % n, cs_v);
      proc.P.send_to((my_num + 1) % n, cs_snew);
    }
  }

  template <class U>
  void change_domain(const vector<int> &regs, int reg_size, U &proc);
};

#endif /* PROTOCOLS_REPLICATED_H_ */
