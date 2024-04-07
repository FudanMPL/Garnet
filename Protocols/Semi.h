/*
 * Semi2k.h
 *
 */

#ifndef PROTOCOLS_SEMI_H_
#define PROTOCOLS_SEMI_H_

#include "SPDZ.h"
#include "Processor/TruncPrTuple.h"

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
  void psi_align(const vector<typename T::clear> &source, const Instruction &instruction, U &proc)
  {
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
