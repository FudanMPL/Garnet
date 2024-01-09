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
template<class T>
class Semi : public SPDZ<T>
{
    SeededPRNG G;

public:
    Semi(Player& P) :
            SPDZ<T>(P)
    {
    }

    void randoms(T& res, int n_bits)
    {
        res.randomize_part(G, n_bits);
    }

    void trunc_pr(const vector<int>& regs, int size,
            SubProcessor<T>& proc)
    {
        trunc_pr(regs, size, proc, T::clear::characteristic_two);
    }

    template<int = 0>
    void trunc_pr(const vector<int>&, int, SubProcessor<T>&, true_type)
    {
        throw not_implemented();
    }

    template<int = 0>
    void trunc_pr(const vector<int>& regs, int size,
            SubProcessor<T>& proc, false_type)
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

        for (auto& info : infos)
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
                                    "during compilation using '-R "
                                    + to_string(min_size) + "'.");
                }
                else
                    throw runtime_error("bit length too large");
            }
            if (this->P.my_num())
                for (int i = 0; i < size; i++)
                    proc.get_S_ref(info.dest_base + i) = -open_type(
                            -open_type(proc.get_S()[info.source_base + i])
                                    >> info.m);
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
  void change_domain(const vector<int>& regs, int reg_size, U& proc){
    assert(regs.size() % 4 == 0);
    assert(proc.P.num_players() == 2);
    assert(proc.Proc != 0);
    typedef typename T::clear value_type;
    typedef typename T::clear bit_type;

    int n = regs.size() / 4;
    int ring_bit_length =  regs[2];

    vector<T> dabits;
    vector<typename T::bit_type> bits;
    vector<bit_type> lsbs_mask_0;
    vector<bit_type> lsbs_mask_1;
    dabits.resize(n * reg_size);
    bits.resize(n * reg_size);
    lsbs_mask_0.resize(n * reg_size);
    lsbs_mask_1.resize(n * reg_size);
    for (int i = 0; i < n * reg_size; i++){
      proc.DataF.get_dabit_no_count(dabits[i], bits[i]);
    }
    if (this->P.my_num() == 0){
      octetStream cs;
      for (int i = 0; i < n; i++){
        for (int k = 0; k < reg_size; k++){
          value_type d0 = proc.S[regs[4 * i + 1] + k];
          proc.input.add_mine(d0);
          value_type overflow_0 = d0 >> (ring_bit_length - 1);
          proc.input.add_mine(overflow_0);
          lsbs_mask_0[i * reg_size + k] = (bit_type) (overflow_0 & 0x1) ^ bits[i * reg_size + k];
          lsbs_mask_0[i * reg_size + k].pack(cs);
        }
      }
      this->P.send_to(1, cs);
      octetStream cs1;

      this->P.receive_player(1, cs1);
      for (int i = 0; i < n * reg_size; i++){
        lsbs_mask_1[i] = cs1.get<bit_type>();
      }
    }
    if (this->P.my_num() == 1){

      octetStream cs;

      for (int i = 0; i < n; i++){
        for (int k = 0; k < reg_size; k++){
          value_type d1 = proc.S[regs[4 * i + 1] + k];
          proc.input.add_mine(d1);

          value_type overflow_1 = -((-d1).arith_right_shift( ring_bit_length - 1));;
          proc.input.add_mine(overflow_1);

          lsbs_mask_1[i * reg_size + k] = (bit_type) (overflow_1 & 0x1) ^ bits[i * reg_size + k];
          lsbs_mask_1[i * reg_size + k].pack(cs);
        }
      }
      this->P.send_to(0, cs);
      octetStream cs0;

      this->P.receive_player(0, cs0);
      for (int i = 0; i < n * reg_size; i++){
        lsbs_mask_0[i] = cs0.get<bit_type>();
      }
    }

    proc.input.add_other(0);
    proc.input.add_other(1);
    proc.input.exchange();

    value_type size(1);
    size = size << (ring_bit_length - 1);

    for (int i = 0; i < n; i++) {
      for (int k = 0; k < reg_size; k++) {
        auto d0 = proc.input.finalize(0);
        auto overflow_0 = proc.input.finalize(0);
        auto d1 = proc.input.finalize(1);
        auto overflow_1 = proc.input.finalize(1);
        auto lsb_mask = lsbs_mask_0[i * reg_size + k] ^ lsbs_mask_1[i * reg_size + k];
        auto lsb = dabits[i * reg_size + k]  -  dabits[i * reg_size + k] * 2 * lsb_mask;
        if (this->P.my_num() == 0)
          lsb = lsb + lsb_mask;

        proc.S[regs[4 * i] + k] = d0 + d1 - (overflow_0 + overflow_1 - lsb) * size;
      }
    }
    }
};

#endif /* PROTOCOLS_SEMI_H_ */
