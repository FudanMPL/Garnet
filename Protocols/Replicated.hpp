/*
 * Replicated.cpp
 *
 */

#ifndef PROTOCOLS_REPLICATED_HPP_
#define PROTOCOLS_REPLICATED_HPP_

#include "Replicated.h"
#include "Processor/Processor.h"
#include "Processor/TruncPrTuple.h"
#include "Tools/benchmarking.h"
#include "Tools/Bundle.h"

#include "ReplicatedInput.h"
#include "Rep3Share2k.h"

#include "ReplicatedPO.hpp"
#include "Math/Z2k.hpp"

template<class T>
ProtocolBase<T>::ProtocolBase() :
        trunc_pr_counter(0), rounds(0), trunc_rounds(0), dot_counter(0),
        bit_counter(0), counter(0)
{
}

template<class T>
Replicated<T>::Replicated(Player& P) : ReplicatedBase(P)
{
    assert(T::vector_length == 2);
}

template<class T>
Replicated<T>::Replicated(const ReplicatedBase& other) :
        ReplicatedBase(other)
{
}

inline ReplicatedBase::ReplicatedBase(Player& P) : P(P)
{
    assert(P.num_players() == 3);
	if (not P.is_encrypted())
		insecure("unencrypted communication", false);

	shared_prngs[0].ReSeed();
	octetStream os;
	os.append(shared_prngs[0].get_seed(), SEED_SIZE);
	P.send_relative(1, os);
	P.receive_relative(-1, os);
	shared_prngs[1].SetSeed(os.get_data());
}

inline ReplicatedBase::ReplicatedBase(Player& P, array<PRNG, 2>& prngs) :
        P(P)
{
    for (int i = 0; i < 2; i++)
        shared_prngs[i].SetSeed(prngs[i]);
}

inline ReplicatedBase ReplicatedBase::branch()
{
    return {P, shared_prngs};
}

template<class T>
ProtocolBase<T>::~ProtocolBase()
{
#ifdef VERBOSE_COUNT
    if (counter or rounds)
        cerr << "Number of " << T::type_string() << " multiplications: "
                << counter << " (" << bit_counter << " bits) in " << rounds
                << " rounds" << endl;
    if (counter or rounds)
        cerr << "Number of " << T::type_string() << " dot products: " << dot_counter << endl;
    if (trunc_pr_counter or trunc_rounds)
        cerr << "Number of probabilistic truncations: " << trunc_pr_counter << " in " << trunc_rounds << " rounds" << endl;
#endif
}

template<class T>
void ProtocolBase<T>::muls(const vector<int>& reg,
        SubProcessor<T>& proc, typename T::MAC_Check& MC, int size)
{
    (void)MC;
    proc.muls(reg, size);
}

template<class T>
void ProtocolBase<T>::mulrs(const vector<int>& reg,
        SubProcessor<T>& proc)
{
    proc.mulrs(reg);
}

template<class T>
void ProtocolBase<T>::multiply(vector<T>& products,
        vector<pair<T, T> >& multiplicands, int begin, int end,
        SubProcessor<T>& proc)
{
#ifdef VERBOSE_CENTRAL
    fprintf(stderr, "multiply from %d to %d in %d\n", begin, end,
            BaseMachine::thread_num);
#endif

    init(proc.DataF, proc.MC);
    init_mul();
    for (int i = begin; i < end; i++)
        prepare_mul(multiplicands[i].first, multiplicands[i].second);
    exchange();
    for (int i = begin; i < end; i++)
        products[i] = finalize_mul();
}

template<class T>
T ProtocolBase<T>::mul(const T& x, const T& y)
{
    init_mul();
    prepare_mul(x, y);
    exchange();
    return finalize_mul();
}

template<class T>
void ProtocolBase<T>::prepare_mult(const T& x, const T& y, int n,
		bool)
{
    prepare_mul(x, y, n);
}

template<class T>
void ProtocolBase<T>::finalize_mult(T& res, int n)
{
    res = finalize_mul(n);
}

template<class T>
T ProtocolBase<T>::finalize_dotprod(int length)
{
    counter += length;
    dot_counter++;
    T res;
    for (int i = 0; i < length; i++)
        res += finalize_mul();
    return res;
}

template<class T>
T ProtocolBase<T>::get_random()
{
    if (random.empty())
    {
        buffer_random();
        assert(not random.empty());
    }

    auto res = random.back();
    random.pop_back();
    return res;
}

template<class T>
vector<int> ProtocolBase<T>::get_relevant_players()
{
    vector<int> res;
    int n = dynamic_cast<typename T::Protocol&>(*this).P.num_players();
    for (int i = 0; i < T::threshold(n) + 1; i++)
        res.push_back(i);
    return res;
}

template<class T>
void Replicated<T>::init_mul()
{
    for (auto& o : os)
        o.reset_write_head();
    add_shares.clear();
}

template<class T>
void Replicated<T>::prepare_mul(const T& x,
        const T& y, int n)
{
    typename T::value_type add_share = x.local_mul(y);
    prepare_reshare(add_share, n);
}

template<class T>
void Replicated<T>::prepare_reshare(const typename T::clear& share,
        int n)
{
    typename T::value_type tmp[2];
    for (int i = 0; i < 2; i++)
        tmp[i].randomize(shared_prngs[i], n);
    auto add_share = share + tmp[0] - tmp[1];
    add_share.pack(os[0], n);
    add_shares.push_back(add_share);
}

template<class T>
void Replicated<T>::exchange()
{
    if (os[0].get_length() > 0)
        P.pass_around(os[0], os[1], 1);
    this->rounds++;
}

template<class T>
void Replicated<T>::start_exchange()
{
    P.send_relative(1, os[0]);
    this->rounds++;
}

template<class T>
void Replicated<T>::stop_exchange()
{
    P.receive_relative(-1, os[1]);
}

template<class T>
inline T Replicated<T>::finalize_mul(int n)
{
    this->counter++;
    this->bit_counter += n;
    T result;
    result[0] = add_shares.next();
    result[1].unpack(os[1], n);
    return result;
}

template<class T>
inline void Replicated<T>::init_dotprod()
{
    init_mul();
    dotprod_share.assign_zero();
}

template<class T>
inline void Replicated<T>::prepare_dotprod(const T& x, const T& y)
{
    dotprod_share = dotprod_share.lazy_add(x.local_mul(y));
}

template<class T>
inline void Replicated<T>::next_dotprod()
{
    dotprod_share.normalize();
    prepare_reshare(dotprod_share);
    dotprod_share.assign_zero();
}

template<class T>
inline T Replicated<T>::finalize_dotprod(int length)
{
    (void) length;
    this->dot_counter++;
    return finalize_mul();
}

template<class T>
T Replicated<T>::get_random()
{
    T res;
    for (int i = 0; i < 2; i++)
        res[i].randomize(shared_prngs[i]);
    return res;
}

template<class T>
void ProtocolBase<T>::randoms_inst(vector<T>& S,
		const Instruction& instruction)
{
    for (int j = 0; j < instruction.get_size(); j++)
    {
        auto& res = S[instruction.get_r(0) + j];
        randoms(res, instruction.get_n());
    }
}

template<class T>
void Replicated<T>::randoms(T& res, int n_bits)
{
    for (int i = 0; i < 2; i++)
        res[i].randomize_part(shared_prngs[i], n_bits);
}

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
        false_type)
{
    assert(regs.size() % 4 == 0);
    assert(proc.P.num_players() == 3);
    assert(proc.Proc != 0);
    typedef typename T::clear value_type;
    int gen_player = 2;
    int comp_player = 1;
    bool generate = P.my_num() == gen_player;
    bool compute = P.my_num() == comp_player;
    ArgList<TruncPrTupleWithGap<value_type>> infos(regs);
    auto& S = proc.get_S();

    octetStream cs;
    ReplicatedInput<T> input(P);

    if (generate)
    {
        SeededPRNG G;
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto r = G.get<value_type>();
                input.add_mine(info.upper(r));
                if (info.small_gap())
                    input.add_mine(info.msb(r));
                (r + S[info.source_base + i][0]).pack(cs);
            }
        P.send_to(comp_player, cs);
    }
    else
        input.add_other(gen_player);

    if (compute)
    {
        P.receive_player(gen_player, cs);
        for (auto info : infos)
            for (int i = 0; i < size; i++)
            {
                auto c = cs.get<value_type>() + S[info.source_base + i].sum();
                input.add_mine(info.upper(c));
                if (info.small_gap())
                    input.add_mine(info.msb(c));
            }
    }

    input.add_other(comp_player);
    input.exchange();
    init_mul();

    for (auto info : infos)
        for (int i = 0; i < size; i++)
        {
            this->trunc_pr_counter++;
            auto c_prime = input.finalize(comp_player);
            auto r_prime = input.finalize(gen_player);
            S[info.dest_base + i] = c_prime - r_prime;

            if (info.small_gap())
            {
                auto c_dprime = input.finalize(comp_player);
                auto r_msb = input.finalize(gen_player);
                S[info.dest_base + i] += ((r_msb + c_dprime)
                        << (info.k - info.m));
                prepare_mul(r_msb, c_dprime);
            }
        }

    exchange();

    for (auto info : infos)
        for (int i = 0; i < size; i++)
            if (info.small_gap())
                S[info.dest_base + i] -= finalize_mul()
                        << (info.k - info.m + 1);
}

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size, U& proc,
        true_type)
{
    (void) regs, (void) size, (void) proc;
    throw runtime_error("trunc_pr not implemented");
}

template<class T>
template<class U>
void Replicated<T>::trunc_pr(const vector<int>& regs, int size,
        U& proc)
{
    this->trunc_rounds++;
    trunc_pr(regs, size, proc, T::clear::characteristic_two);
}

//template<class T>
//template<class U>
//void Replicated<T>::change_domain(const vector<int>& regs, U& proc){
//  assert(regs.size() % 3 == 0);
//  assert(proc.P.num_players() == 3);
//  assert(proc.Proc != 0);
//  typedef typename T::clear value_type;
//  ReplicatedInput<T> input(P);
//  int n = regs.size() / 3;
//  int small_ring_size =  regs[2];
//  if (P.my_num() == 0){
//    for (int i = 0; i < n; i++){
//      value_type temp = proc.S[regs[3 * i + 1]].sum();
//      value_type temp_0 = temp - ((temp >> small_ring_size) << small_ring_size);
//      auto a0 = temp_0 >> small_ring_size;
//      auto b0 = temp_0 >> (small_ring_size - 1);
//      cout << "temp_0= " << temp_0 << endl;
//      cout << "a0= " << a0  << endl;
//      cout << "b0= " << b0  << endl;
//      input.add_mine(temp_0);
//      input.add_mine(a0);
//      input.add_mine(b0);
//    }
//  }
//  if (P.my_num() == 1){
//    for (int i = 0; i < n; i++){
//      value_type temp = proc.S[regs[3 * i + 1]][0];
//      value_type temp_1 = temp - ((temp >> small_ring_size) << small_ring_size);
//      auto a1 = -((-temp_1).arith_right_shift(small_ring_size));
//      auto b1 = -((-temp_1).arith_right_shift(small_ring_size - 1));
//      cout << "temp_1= " << temp_1 << endl;
//      cout << "a1= " << a1  << endl;
//      cout << "b1= " << b1  << endl;
//      input.add_mine(temp_1);
//      input.add_mine(a1);
//      input.add_mine(b1);
//    }
//  }
//  input.add_other(0);
//  input.add_other(1);
//  input.exchange();
//
//  value_type size(1);
//  size = size << (small_ring_size );
//  for (int i = 0; i < n; i++){
//    auto temp_0 = input.finalize(0);
//    auto a0 = input.finalize(0);
//    auto b0 = input.finalize(0);
//    auto temp_1 = input.finalize(1);
//    auto a1 = input.finalize(1);
//    auto b1 = input.finalize(1);
//    cout << b0 << b1 << endl;
//    proc.S[regs[3 * i]] = temp_0 + temp_1 - (a0 + a1 )  * size;
//  }
//}

//template<class T>
//template<class U>
//void Replicated<T>::change_domain(const vector<int>& regs, int reg_size, U& proc){
//  assert(regs.size() % 4 == 0);
//  assert(proc.P.num_players() == 3);
//  assert(proc.Proc != 0);
//  typedef typename T::clear value_type;
//  ReplicatedInput<T> input(P);
//  int n = regs.size() / 4;
//  int ring_bit_length =  regs[2];
//  value_type size(1);
//  value_type half_size(1);
//  size = size << ring_bit_length;
//  half_size = half_size << (ring_bit_length - 1);
//  if (P.my_num() == 0){
//    octetStream cs;
//    for (int i = 0; i < n; i++){
//      for (int k = 0; k < reg_size; k++){
//        value_type temp = proc.S[regs[4 * i + 1] + k].sum();
//        temp = temp - ((temp >> ring_bit_length) << ring_bit_length);
//        input.add_mine(temp);
//
//        value_type temp_0 = temp + half_size;
//        auto wrap0 = temp_0 >> ring_bit_length;
//        input.add_mine(wrap0);
//        temp_0 = temp_0 - ((temp_0 >> ring_bit_length) << ring_bit_length);
//        temp_0 = temp_0 - size;
//        // todo: multiply a random coin
//        int lx = regs[4 * i + 3];
//        value_type truncs[lx];
//        for (int j=0; j < lx; j++){
//          truncs[j] = temp_0.arith_right_shift(j + ring_bit_length - 1);
//        }
//        // todo: shuffle and multiply random
//        for (int j=0; j < lx; j++){
//          truncs[j].pack(cs);
//        }
//      }
//    }
//    P.send_to(2, cs);
//  }
//  if (P.my_num() == 1){
//    octetStream cs;
//    for (int i = 0; i < n; i++){
//      for (int k = 0; k < reg_size; k++){
//        value_type temp = proc.S[regs[4 * i + 1] + k][0];
//        value_type temp_1 = temp - ((temp >> ring_bit_length) << ring_bit_length);
//        input.add_mine(temp_1);
//        // todo: multiply a random coin
//        int lx = regs[4 * i + 3];
//        value_type truncs[lx];
//        for (int j=0; j < lx; j++){
//          truncs[j] = -((-temp_1).arith_right_shift(j + ring_bit_length - 1));
//        }
//        // todo: shuffle and multiply random
//        for (int j=0; j < lx; j++){
//          truncs[j].pack(cs);
//        }
//      }
//
//    }
//    P.send_to(2, cs);
//  }
//
//  if (P.my_num() == 2){
//    octetStream cs0;
//    octetStream cs1;
//    P.receive_player(0, cs0);
//    P.receive_player(1, cs1);
//    for (int i = 0; i < n; i++) {
//      for (int k = 0; k < reg_size; k++) {
//        int lx = regs[4 * i + 3];
//        value_type truncs[lx];
//        value_type wrap2(1);
//        for (int j = 0; j < lx; j++) {
//          truncs[j] = cs0.get<value_type>() + cs1.get<value_type>();
//          if (truncs[j] == 1) {
//            wrap2 = 1;
//          }
//          if (truncs[j] == -1) {
//            wrap2 = 0;
//          }
//        }
//        input.add_mine(wrap2);
//      }
//    }
//  }
//  input.add_other(0);
//  input.add_other(1);
//  input.add_other(2);
//  input.exchange();
//
//
//  for (int i = 0; i < n; i++) {
//    for (int k = 0; k < reg_size; k++) {
//      auto temp_0 = input.finalize(0);
//      auto wrap0 = input.finalize(0);
//      auto temp_1 = input.finalize(1);
//      auto wrap1 = input.finalize(2);
//      proc.S[regs[4 * i] + k] = temp_0 + temp_1 - (wrap0 + wrap1) * size;
//    }
//  }
//}

template<class T>
template<class U>
void Replicated<T>::change_domain(const vector<int>& regs, int reg_size, U& proc){
  assert(regs.size() % 4 == 0);
  assert(proc.P.num_players() == 3);
  assert(proc.Proc != 0);
  typedef typename T::clear value_type;
  typedef typename T::clear bit_type;
  ReplicatedInput<T> input(P);

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
  if (P.my_num() == 0){
    octetStream cs;
    octetStream cs_2;
    for (int i = 0; i < n; i++){
      for (int k = 0; k < reg_size; k++){
        value_type temp = proc.S[regs[4 * i + 1] + k].sum();
        temp = temp - ((temp >> ring_bit_length) << ring_bit_length);
        input.add_mine(temp);
        value_type overflow_0 = temp >> (ring_bit_length - 1);
        input.add_mine(overflow_0);

        lsbs_mask_0[i * reg_size + k] = (bit_type) (overflow_0 & 0x1) ^ bits[i * reg_size + k][0] ^ bits[i * reg_size + k][1];
        lsbs_mask_0[i * reg_size + k].pack(cs);
        lsbs_mask_0[i * reg_size + k].pack(cs_2);
        }
    }
    P.send_to(1, cs);
    P.send_to(2, cs_2);
    octetStream cs1;

    P.receive_player(1, cs1);
    for (int i = 0; i < n * reg_size; i++){
      lsbs_mask_1[i] = cs1.get<bit_type>();
    }
  }
  if (P.my_num() == 1){

    octetStream cs;
    octetStream cs_2;
    for (int i = 0; i < n; i++){
      for (int k = 0; k < reg_size; k++){
        value_type temp = proc.S[regs[4 * i + 1] + k][0];
        input.add_mine(temp);

        value_type overflow_1 = -((-temp).arith_right_shift( ring_bit_length - 1));;
        input.add_mine(overflow_1);

        lsbs_mask_1[i * reg_size + k] = (bit_type) (overflow_1 & 0x1) ^ bits[i * reg_size + k][0];
        lsbs_mask_1[i * reg_size + k].pack(cs);
        lsbs_mask_1[i * reg_size + k].pack(cs_2);
      }
    }
    P.send_to(0, cs);
    P.send_to(2, cs_2);
    octetStream cs0;

    P.receive_player(0, cs0);
    for (int i = 0; i < n * reg_size; i++){
      lsbs_mask_0[i] = cs0.get<bit_type>();
    }
  }
  if (P.my_num() == 2){

    octetStream cs0;
    octetStream cs1;
    P.receive_player(0, cs0);
    P.receive_player(1, cs1);
    for (int i = 0; i < n * reg_size; i++){
      lsbs_mask_0[i] = cs0.get<bit_type>();
      lsbs_mask_1[i] = cs1.get<bit_type>();
    }
  }

  input.add_other(0);
  input.add_other(1);
  input.exchange();

  value_type size(1);
  size = size << (ring_bit_length - 1);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < reg_size; k++) {
      auto temp_0 = input.finalize(0);
      auto overflow_0 = input.finalize(0);
      auto temp_1 = input.finalize(1);
      auto overflow_1 = input.finalize(1);
      auto lsb_mask = lsbs_mask_0[i * reg_size + k] ^ lsbs_mask_1[i * reg_size + k];

      auto lsb = dabits[i * reg_size + k]  -  dabits[i * reg_size + k] * 2 * lsb_mask;
      if (P.my_num() < 2)
        lsb[P.my_num()] = lsb[P.my_num()] + lsb_mask;

      proc.S[regs[4 * i] + k] = temp_0 + temp_1 - (overflow_0 + overflow_1 - lsb) * size;
    }
  }
}

#endif
