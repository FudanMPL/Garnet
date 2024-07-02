/*
 * Semi2k.h
 *
 */

#ifndef PROTOCOLS_SEMI_H_
#define PROTOCOLS_SEMI_H_

#include "SPDZ.h"
#include "Processor/TruncPrTuple.h"

#include "Tools_PSI/SimpleIndex.h"
#include "cryptoTools/Common/CuckooIndex.h"
#include "OT/OTExtension.h"
#include "OT/OTExtensionWithMatrix.h"
#include "Tools/octetStream.h"

#define RECEIVER_P 0

typedef uint64_t idtype;

/**
 * Dishonest-majority protocol for computation modulo a power of two
 */
template <class T>
class Semi : public SPDZ<T>
{
  SeededPRNG G;

public:
  Semi(Player &P) : SPDZ<T>(P)
  {
  }

  void randoms(T &res, int n_bits)
  {
    res.randomize_part(G, n_bits);
  }

  void trunc_pr(const vector<int> &regs, int size,
                SubProcessor<T> &proc)
  {
    trunc_pr(regs, size, proc, T::clear::characteristic_two);
  }

  template <int = 0>
  void trunc_pr(const vector<int> &, int, SubProcessor<T> &, true_type)
  {
    throw not_implemented();
  }

  template <int = 0>
  void trunc_pr(const vector<int> &regs, int size,
                SubProcessor<T> &proc, false_type)
  {
    if (this->P.num_players() > 2)
      throw runtime_error("probabilistic truncation "
                          "only implemented for two players");

    assert(regs.size() % 4 == 0);
    this->trunc_pr_counter += size * regs.size() / 4;
    typedef typename T::open_type open_type;

    vector<TruncPrTupleWithGap<open_type>> infos;
    for (size_t i = 0; i < regs.size(); i += 4)
      infos.push_back({regs, i});

    for (auto &info : infos)
    {
      if (not info.big_gap())
      {
        if (not T::clear::invertible)
        {
          int min_size = 64 * DIV_CEIL(
                                  info.k + OnlineOptions::singleton.trunc_error, 64);
          throw runtime_error(
              "Bit length too large for trunc_pr. "
              "Disable it or increase the ring size "
              "during compilation using '-R " +
              to_string(min_size) + "'.");
        }
        else
          throw runtime_error("bit length too large");
      }
      if (this->P.my_num())
        for (int i = 0; i < size; i++)
          proc.get_S_ref(info.dest_base + i) = -open_type(
              -open_type(proc.get_S()[info.source_base + i]) >> info.m);
      else
        for (int i = 0; i < size; i++)
          proc.get_S_ref(info.dest_base + i) =
              proc.get_S()[info.source_base + i] >> info.m;
    }
  }

  void buffer_random()
  {
    for (int i = 0; i < OnlineOptions::singleton.batch_size; i++)
      this->random.push_back(G.get<T>());
  }

  template <class U>
  void psi(const vector<typename T::clear> &source, const Instruction &instruction, U &proc)
  {
    // if constexpr (typeid(T) == typeid(SemiShare))
    // {
    //   throw not_implemented();
    // }
    // octetStream cs;
    // int r0 = instruction.get_r(0);
#ifdef ENABLE_PSI
    octetStream cs0, cs, cs2;
    auto res = proc.C.begin() + (instruction.get_r(0));

    // typename T::clear result;
    auto &args = instruction.get_start();
    fstream r;
    idtype m = args.back();
    // cout << args.size() << " m: " << m << endl;
    // for (size_t i = 0; i < args.size(); i++)
    // {
    //   cout << args[i] << " ";
    // }
    // cout << endl;

    r.open("Player-Data/PSI/ID-P" + to_string(proc.P.my_num()), ios::in);
    vector<osuCrypto::block> ids;
    vector<idtype> smallids;
    idtype r_tmp;
    for (size_t i = 0; i < m; i++)
    {
      r >> r_tmp;
      smallids.push_back(r_tmp);
      ids.push_back(osuCrypto::block(r_tmp));
    }
    r.close();
    // for (size_t i = 0; i < m; i++)
    // {
    //   cout << ids[i] << " ";
    // }
    // cout << endl;

    // ssp 40
    int ssp = 40;
    osuCrypto::CuckooParam params = oc::CuckooIndex<>::selectParams(m, ssp, 0, 3);
    int nbase = 128;
    size_t l = sizeof(idtype) * 8;
    int nOTs = l * params.numBins();
    // PRNG G;
    // G.ReSeed();
    OT_ROLE ot_role;
    // cout << params.numBins() << endl;
    osuCrypto::CuckooIndex<> cuckoo;
    SimpleIndex sIdx;
    if (proc.P.my_num() == RECEIVER_P)
    {
      int seed = 0;
      cs0.store(seed);
      proc.P.send_to(1 - RECEIVER_P, cs0);
      osuCrypto::block cuckooSeed(seed);
      cuckoo.init(params);
      cuckoo.insert(ids, cuckooSeed);
      // cuckoo.print();
      ot_role = RECEIVER;
    }
    else
    {
      proc.P.receive_player(RECEIVER_P, cs0);
      int seed;
      cs0.get(seed);
      osuCrypto::block cuckooSeed(seed);
      sIdx.init(params.numBins(), m, ssp, 3);
      sIdx.insertItems(ids, cuckooSeed);
      // sIdx.print();
      ot_role = SENDER;
    }

    // cout << "begin base ot \n";
    // base ot
    timeval baseOTstart, baseOTend;
    gettimeofday(&baseOTstart, NULL);
    RealTwoPartyPlayer *rP = new RealTwoPartyPlayer(proc.P.N, 1 - proc.P.my_num(), "machine");
    BaseOT bot = BaseOT(nbase, 128, rP, INV_ROLE(ot_role));
    bot.exec_base();
    gettimeofday(&baseOTend, NULL);
    double basetime = timeval_diff(&baseOTstart, &baseOTend);
    // cout << "BaseTime (" << role_to_str(ot_role) << "): " << basetime / 1000000 << endl
    //      << flush;
    // Receiver send something to force synchronization
    // (since Sender finishes baseOTs before Receiver)
    if (proc.P.my_num() == RECEIVER_P)
    {
      bigint a = 3;
      a.pack(cs0);
      proc.P.send_to(1 - RECEIVER_P, cs0);
    }
    else
    {
      proc.P.receive_player(RECEIVER_P, cs0);
      bigint a;
      a.unpack(cs0);
      // cout << a << endl;
    }

    // convert baseOT selection bits to BitVector
    // (not already BitVector due to legacy PVW code)
    BitVector baseReceiverInput = bot.receiver_inputs;
    baseReceiverInput.resize(nbase);

    OTExtensionWithMatrix *ot_ext = new OTExtensionWithMatrix(rP, ot_role);
    BitVector receiverInput(nOTs);
    if (proc.P.my_num() == RECEIVER_P)
    {
      idtype idx;
      for (size_t i = 0; i < params.numBins(); i++)
      {
        if (!cuckoo.mBins[i].isEmpty())
        {
          idx = smallids[cuckoo.mBins[i].idx()];
          // cout << idx << " | ";
          receiverInput.set_word(i, idx);
          // cout << receiverInput.get_word(i) << endl;
        }
      }
      // cout << receiverInput.str() << endl;
      // receiverInput.randomize(G);
    }
    // cout << receiverInput.str() << flush;
    // cout << "Running " << nOTs << " OT extensions\n"
    //      << flush;

    // cout << "Initialize OT Extension\n";
    timeval OTextstart, OTextend;
    gettimeofday(&OTextstart, NULL);

    ot_ext->init(baseReceiverInput,
                 bot.sender_inputs, bot.receiver_outputs);
    ot_ext->transfer(nOTs, receiverInput, 1);
    // ot_ext.check();
    bot.extend_length();

    // print
    // for (int i = 0; i < nOTs; i++)
    // {
    //   if (ot_role == SENDER)
    //   {
    //     // send both inputs over
    //     cout << i << " " << bot.sender_inputs[i][0].str() << " | " << bot.sender_inputs[i][1].str() << endl;
    //   }
    //   else
    //   {
    //     cout << i << " " << receiverInput[i] << ": " << bot.receiver_outputs[i].str() << endl;
    //   }
    // }
    // bot.check();

    gettimeofday(&OTextend, NULL);
    double totaltime = timeval_diff(&OTextstart, &OTextend);
    // cout << "Time for OTExt (" << role_to_str(ot_role) << "): " << totaltime / 1000000 << endl
    //      << flush;

    // caculate oprf
    idtype num;
    vector<idtype> inter_ids;
    if (proc.P.my_num() == RECEIVER_P)
    {
      // vector<BitVector> r_fs(params.numBins());
      // receive oprf result
      proc.P.receive_player(1 - RECEIVER_P, cs);
      vector<string> s_fs;
      idtype mm = m * 3;
      BitVector f_temp;
      for (size_t i = 0; i < mm; i++)
      {
        f_temp.unpack(cs);
        s_fs.push_back(f_temp.str());
        // cout << f_temp.str() << endl;
      }
      sort(s_fs.begin(), s_fs.end());

      // compare to find intersection set
      BitVector key, temp;
      string strkey;
      key.resize(sizeof(__m128i) << 3);
      idtype inter_id;
      for (unsigned int i = 0; i < params.numBins(); i++)
      {
        if (!cuckoo.mBins[i].isEmpty())
        {
          // cout << i << ": ";
          key.assign_zero();
          // get oprf
          for (unsigned int j = 0; j < l; j++)
          {
            temp.assign_bytes((char *)ot_ext->get_receiver_output(i * l + j), sizeof(__m128i));
            // cout << temp.str() << endl;
            key.add(temp);
          }
          // r_fs[i] = key;
          // find same element
          strkey = key.str();
          // cout << strkey << endl;
          bool found = binary_search(s_fs.begin(), s_fs.end(), strkey);
          if (found)
          {
            inter_id = smallids[cuckoo.mBins[i].idx()];
            // cout << inter_id << ": " << strkey << endl;
            inter_ids.push_back(inter_id);
          }
          // cout << smallids[cuckoo.mBins[i].idx()] << " " << r_fs[i].str() << endl;
        }
      }
      sort(inter_ids.begin(), inter_ids.end());
      num = inter_ids.size();
      // proc.Proc->public_file << num << "\n";
      cs2.store(num);
      for (const idtype &inter_id : inter_ids)
      {
        cs2.store(inter_id);
        // cout << inter_id << endl;
        // proc.Proc->public_file << inter_id << "\n";
        // proc.Proc->public_output << inter_id << "\n";
      }
      proc.P.send_to(1 - RECEIVER_P, cs2);
      // open result to sender
    }
    else // sender
    {
      BitVector key, temp;
      idtype id;
      vector<BitVector> fs;
      // vector<vector<BitVector>> fkx(params.numBins());
      key.resize(sizeof(__m128i) << 3);
      for (unsigned int i = 0; i < params.numBins(); i++)
      {
        // cout << "Bin #" << i << endl;
        array<vector<BitVector>, 2> outs;
        ot_ext->get_sender_output128i(outs, i * l, l);

        // for (size_t jj = i * params.numBins(); jj < i * l + l; jj++)
        // {
        //   cout << outs[0][jj].str() << " " << outs[1][jj].str() << endl;
        // }

        for (unsigned int k = 0; k < sIdx.mBinSizes[i]; k++)
        {
          key.assign_zero();
          id = smallids[sIdx.mBins(i, k).idx()];
          // cout << id << " | ";
          for (unsigned int j = 0; j < l; j++)
          {
            // cout << (id & 0x1);
            temp = outs[id & 0x1][j];
            id = id >> 1;
            key.add(temp);
          }
          fs.push_back(key);
          // cout << key.str() << endl;
          // cout << "|   " << fkx[i][k].str() << endl;
        }
      }
      for (BitVector fk : fs)
      {
        fk.pack(cs);
      }
      proc.P.send_to(RECEIVER_P, cs);
      proc.P.receive_player(RECEIVER_P, cs2);
      idtype inter_id;
      cs2.get(num);
      // proc.Proc->public_file << num << "\n";
      for (size_t i = 0; i < num; i++)
      {
        cs2.get(inter_id);
        inter_ids.push_back(inter_id);
        // proc.Proc->public_file << inter_id << "\n";
        // cout << inter_id << endl;
        // dest[i] = inter_id;
      }
    }
    // proc.Proc->public_file.seekg(0);

    res[0] = num;
    for (size_t i = 0; i < num; i++)
    {
      res[i + 1] = inter_ids[i];
      // cout << inter_ids[i] << endl;
    }
    // delete bot;
    delete ot_ext;

    if (0)
    {
      BitVector receiver_output, sender_output;
      // char filename[1024];
      // sprintf(filename, RECEIVER_INPUT, P.my_num());
      // ofstream outf(filename);
      // receiverInput.output(outf, false);
      // outf.close();
      // sprintf(filename, RECEIVER_OUTPUT, P.my_num());
      // outf.open(filename);
      if (ot_role == SENDER)
      {

        // outf.close();

        // sprintf(filename, SENDER_OUTPUT, P.my_num(), i);
        // outf.open(filename);
        for (int j = 0; j < nOTs; j++)
        {
          for (int i = 0; i < 2; i++)
          {
            sender_output.assign_bytes((char *)ot_ext->get_sender_output(i, j), sizeof(__m128i));
            cout << sender_output.str() << "  ";
            // sender_output.output(outf, false);
          }
          cout << endl;
          // outf.close();
        }
      }
      else
      {
        for (unsigned int i = 0; i < nOTs; i++)
        {
          receiver_output.assign_bytes((char *)ot_ext->get_receiver_output(i), sizeof(__m128i));
          cout << receiverInput[i] << ": " << receiver_output.str() << endl;
          // receiver_output.output(outf, false);
        }
      }
    }
    // return 0;
#else
    throw not_implemented();
#endif
  }

  template <class U>
  void psi_align(const vector<typename T::clear> &source, const Instruction &instruction, U &proc)
  {
#ifdef ENABLE_PSI
    // cout << "psi_align" << endl;
    typedef uint64_t idtype;
#define RECEIVER_P 0
    auto &dim = instruction.get_start();
    auto res = proc.S.begin() + (instruction.get_r(0));
    auto ids = source.begin() + instruction.get_r(1);

    const void *tmp = (*ids).get_ptr();
    idtype num = *reinterpret_cast<const idtype *>(tmp);
    // cout << "num: " << num << endl;
    ids++;
    idtype n = dim[0];
    // cout << "align " << distance(S.begin(), res) << " " << distance(res, S.end()) << " " << distance(source.begin(), ids) << " " << distance(ids, source.end()) << "\n";
    // for (size_t i = 0; i < num; i++)
    // {
    //   cout << *(ids + i) << " ";
    // }
    // cout << endl;

    // step1: find position of ids
    string idfile = "Player-Data/PSI/ID-P" + to_string(proc.P.my_num());
    // vector<idtype> lines = readIDP(source, idfile);
    ifstream fid;
    fid.open(idfile, ios::in);
    if (!fid.is_open())
    {
      cerr << "Error opening file: " << proc.P.my_num() << endl;
      return;
    }
    idtype id_tmp, l;
    std::vector<idtype> lines(num);
    auto idend = ids + num;
    for (size_t i = 0; i < n; i++)
    {
      // Convert the line to an integer (assuming the file contains integers)
      try
      {
        fid >> id_tmp;
        // myids.push_back(id_tmp);
        auto it = find(ids, idend, id_tmp);
        if (it != idend)
        {
          l = std::distance(ids, it);
          // cout << id_tmp << " " << i << " " << l << endl;
          lines[l] = i;
        }
      }
      catch (const invalid_argument &e)
      {
        // Handle the case where the line is not a valid integer
        cerr << "Invalid data in line " << i << ": " << id_tmp << endl;
      }
    }
    fid.close();

    // step2: find features of ids
    string ffile = "Player-Data/PSI/F-P" + to_string(proc.P.my_num());
    // vector<vector<idtype>> fs = readFeature(ffile, lines, n, dim[P.my_num() + 1]);
    idtype row = dim[0];
    idtype col = dim[1 + proc.P.my_num()];
    vector<string> features;
    ifstream file(ffile);
    if (file.is_open())
    {
      string line;
      idtype currentLine = 0;

      while (getline(file, line) && currentLine < row)
      {
        // std::cout << line << endl;
        features.push_back(line);
        currentLine++;
      }
      file.close();
    }
    else
    {
      cerr << "Unable to open file: " << ffile << endl;
    }

    // idtype currentCol = 0;
    idtype value;
    istringstream iss;
    vector<vector<idtype>> mfs(num, vector<idtype>(col, 0));
    for (size_t i = 0; i < num; i++)
    {
      iss = istringstream(features[lines[i]]);
      for (size_t j = 0; j < col; j++)
      {
        iss >> value;
        mfs[i][j] = value;
        // cout << value << " ";

        // split by ','
        // if (currentCol < col)
        //   iss.ignore(numeric_limits<streamsize>::max(), ',');
      }
      // cout << endl;
    }

    // int lambda = T::clear::MAX_N_BITS;
    // MC->init_open(P, lambda)
    ;
    // step3: exchange fs
    // step3.1: generate randomness
    int cols = dim[1] + dim[2];
    octetStream cs[2];
    octetStream cs0 = cs[proc.P.my_num()];
    octetStream cs1 = cs[1 - proc.P.my_num()];
    vector<vector<T>> fs_rand(num, vector<T>(col, 0)); // save randomness
    T share_tmp;
    for (size_t i = 0; i < num; i++)
    {
      vector<T> rand_line;
      for (size_t j = 0; j < col; j++)
      {
        share_tmp = proc.DataF.get_random();
        share_tmp.pack(cs0);
        fs_rand[i][j] = share_tmp;
        // cout << fs_rand[i][j] << " ";
      }
      // cout << endl;
    }
    proc.P.send_to(1 - proc.P.my_num(), cs0);
    proc.P.receive_player(1 - proc.P.my_num(), cs1);

    T f_tmp;
    // cout << mfs.size() << " " << mfs[0].size() << " " << fs_rand.size() << " " << fs_rand[0].size() << endl;
    if (proc.P.my_num() == RECEIVER_P)
    {
      for (size_t i = 0; i < num; i++)
      {
        // cout << i << ":" << endl;
        for (size_t j = 0; j < col; j++)
        {
          // MC->prepare_open(proc.S[args[i * cols + j]]);
          // cout << j << endl;
          *(res + i * cols + j) = (T)mfs[i][j] - fs_rand[i][j];
        }
        for (int j = dim[1]; j < cols; j++)
        {
          // MC->prepare_open(proc.S[args[i * cols + dim[1] + j]]);
          cs1.get(f_tmp);
          // cout << f_tmp << endl;
          *(res + i * cols + j) = f_tmp;
          // cout << j << " |";
        }
        // cout << endl;
      }
    }
    else
    {
      for (size_t i = 0; i < num; i++)
      {
        for (size_t j = 0; j < dim[1]; j++)
        {
          // MC->prepare_open(proc.S[args[i * cols + j]]);
          f_tmp.unpack(cs1);
          *(res + i * cols + j) = f_tmp;
        }
        for (size_t j = 0; j < col; j++)
        {
          // MC->prepare_open(proc.S[args[i * cols + dim[1] + j]]);
          *(res + i * cols + dim[1] + j) = (T)mfs[i][j] - fs_rand[i][j];
        }
      }
    }
#else
    throw not_implemented();
#endif
  }

  template <class U>
  void change_domain(const vector<int> &regs, int reg_size, U &proc)
  {
    assert(regs.size() % 4 == 0);
    assert(proc.P.num_players() == 2);
    assert(proc.Proc != 0);
    typedef typename T::clear value_type;
    typedef typename T::clear bit_type;

    int n = regs.size() / 4;
    int ring_bit_length = regs[2];

    vector<T> dabits;
    vector<typename T::bit_type> bits;
    vector<bit_type> lsbs_mask_0;
    vector<bit_type> lsbs_mask_1;
    dabits.resize(n * reg_size);
    bits.resize(n * reg_size);
    lsbs_mask_0.resize(n * reg_size);
    lsbs_mask_1.resize(n * reg_size);
    for (int i = 0; i < n * reg_size; i++)
    {
      proc.DataF.get_dabit_no_count(dabits[i], bits[i]);
    }
    if (this->P.my_num() == 0)
    {
      octetStream cs;
      for (int i = 0; i < n; i++)
      {
        for (int k = 0; k < reg_size; k++)
        {
          value_type d0 = proc.S[regs[4 * i + 1] + k];
          proc.input.add_mine(d0);
          value_type overflow_0 = d0 >> (ring_bit_length - 1);
          proc.input.add_mine(overflow_0);
          lsbs_mask_0[i * reg_size + k] = (bit_type)(overflow_0 & 0x1) ^ bits[i * reg_size + k];
          lsbs_mask_0[i * reg_size + k].pack(cs);
        }
      }
      this->P.send_to(1, cs);
      octetStream cs1;

      this->P.receive_player(1, cs1);
      for (int i = 0; i < n * reg_size; i++)
      {
        lsbs_mask_1[i] = cs1.get<bit_type>();
      }
    }
    if (this->P.my_num() == 1)
    {

      octetStream cs;

      for (int i = 0; i < n; i++)
      {
        for (int k = 0; k < reg_size; k++)
        {
          value_type d1 = proc.S[regs[4 * i + 1] + k];
          proc.input.add_mine(d1);

          value_type overflow_1 = -((-d1).arith_right_shift(ring_bit_length - 1));
          ;
          proc.input.add_mine(overflow_1);

          lsbs_mask_1[i * reg_size + k] = (bit_type)(overflow_1 & 0x1) ^ bits[i * reg_size + k];
          lsbs_mask_1[i * reg_size + k].pack(cs);
        }
      }
      this->P.send_to(0, cs);
      octetStream cs0;

      this->P.receive_player(0, cs0);
      for (int i = 0; i < n * reg_size; i++)
      {
        lsbs_mask_0[i] = cs0.get<bit_type>();
      }
    }

    proc.input.add_other(0);
    proc.input.add_other(1);
    proc.input.exchange();

    value_type size(1);
    size = size << (ring_bit_length - 1);

    for (int i = 0; i < n; i++)
    {
      for (int k = 0; k < reg_size; k++)
      {
        auto d0 = proc.input.finalize(0);
        auto overflow_0 = proc.input.finalize(0);
        auto d1 = proc.input.finalize(1);
        auto overflow_1 = proc.input.finalize(1);
        auto lsb_mask = lsbs_mask_0[i * reg_size + k] ^ lsbs_mask_1[i * reg_size + k];
        auto lsb = dabits[i * reg_size + k] - dabits[i * reg_size + k] * 2 * lsb_mask;
        if (this->P.my_num() == 0)
          lsb = lsb + lsb_mask;

        proc.S[regs[4 * i] + k] = d0 + d1 - (overflow_0 + overflow_1 - lsb) * size;
      }
    }
  }
};

#endif /* PROTOCOLS_SEMI_H_ */
