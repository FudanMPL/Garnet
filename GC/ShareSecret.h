/*
 * ReplicatedSecret.h
 *
 */

#ifndef GC_SHARESECRET_H_
#define GC_SHARESECRET_H_

#include <vector>
using namespace std;

#include "GC/Memory.h"
#include "GC/Clear.h"
#include "GC/Access.h"
#include "GC/ArgTuples.h"
#include "GC/NoShare.h"
#include "Math/FixedVec.h"
#include "Math/BitVec.h"
#include "Tools/SwitchableOutput.h"
#include "Protocols/Replicated.h"
#include "Protocols/ReplicatedMC.h"
#include "Processor/DummyProtocol.h"
#include "Processor/ProcessorBase.h"
#include "Processor/Instruction.h"

namespace GC
{

template <class T>
class Processor;

template <class T>
class Thread;

template <class T>
class Machine;

template<class U>
class ShareSecret
{
public:
    typedef U whole_type;

    typedef Memory<U> DynamicMemory;
    typedef SwitchableOutput out_type;

    static const bool is_real = true;
    static const bool actual_inputs = true;

    static ShareThread<U>& get_party()
    {
        return ShareThread<U>::s();
    }

    static void store_clear_in_dynamic(Memory<U>& mem,
            const vector<ClearWriteAccess>& accesses);

    static void load(vector< ReadAccess<U> >& accesses, const Memory<U>& mem);
    static void store(Memory<U>& mem, vector< WriteAccess<U> >& accesses);

    static void andrs(Processor<U>& processor, const vector<int>& args)
    { and_(processor, args, true); }
    static void ands(Processor<U>& processor, const vector<int>& args)
    { and_(processor, args, false); }
    static void and_(Processor<U>& processor, const vector<int>& args, bool repeat);
    static void andrsvec(Processor<U>& processor, const vector<int>& args);
    static void xors(Processor<U>& processor, const vector<int>& args);
    static void inputb(Processor<U>& processor, const vector<int>& args)
    { inputb(processor, processor, args); }
    static void inputb(Processor<U>& processor, ProcessorBase& input_processor,
            const vector<int>& args);
    static void inputbvec(Processor<U>& processor, ProcessorBase& input_processor,
            const vector<int>& args);
    static void reveal_inst(Processor<U>& processor, const vector<int>& args);

    template<class T>
    static void convcbit(Integer& dest, const Clear& source, T&) { dest = source; }

    template<class T>
    static void convcbit2s(Processor<T>& processor, const BaseInstruction& instruction)
    { processor.convcbit2s(instruction); }
    static void andm(Processor<U>& processor, const BaseInstruction& instruction)
    { processor.andm(instruction); }

    static BitVec get_mask(int n) { return n >= 64 ? -1 : ((1L << n) - 1); }

    void check_length(int n, const Integer& x);

    void invert(int n, const U& x);

    void random_bit();

    template<class T>
    void my_input(T& inputter, BitVec value, int n_bits);
    template<class T>
    void other_input(T& inputter, int from, int n_bits = 1);
    template<class T>
    void finalize_input(T& inputter, int from, int n_bits);

    U& operator=(const U&);
};

template<class U, int L>
class RepSecretBase : public FixedVec<BitVec, L>, public ShareSecret<U>
{
    typedef FixedVec<BitVec, L> super;
    typedef RepSecretBase This;

public:
    typedef U part_type;
    typedef U small_type;
    typedef U whole_type;

    typedef BitVec clear;
    typedef BitVec open_type;
    typedef NoShare mac_type;
    typedef NoValue mac_key_type;

    typedef NoShare bit_type;

    static const int N_BITS = clear::N_BITS;

    static const bool dishonest_majority = false;
    static const bool variable_players = false;
    static const bool needs_ot = false;
    static const bool has_mac = false;

    static string type_string() { return "replicated secret"; }
    static string phase_name() { return "Replicated computation"; }

    static const int default_length = N_BITS;

    static int threshold(int)
    {
        return 1;
    }

    static void trans(Processor<U>& processor, int n_outputs,
            const vector<int>& args);

    template<class T>
    static void generate_mac_key(mac_key_type, T)
    {
    }

    static void read_or_generate_mac_key(string, const Player&, mac_key_type)
    {
    }

    RepSecretBase()
    {
    }
    template <class T>
    RepSecretBase(const T& other) :
            super(other)
    {
    }

    void bitcom(Memory<U>& S, const vector<int>& regs);
    void bitdec(Memory<U>& S, const vector<int>& regs) const;

    void xor_(int n, const This& x, const This& y)
    { *this = x ^ y; (void)n; }

    This operator&(const Clear& other)
    { return super::operator&(BitVec(other)); }

    This lsb()
    { return *this & 1; }

    This get_bit(int i)
    { return (*this >> i) & 1; }

    void xor_bit(int i, const This& bit)
    { *this ^= bit << i; }
};

template<class U>
class ReplicatedSecret : public RepSecretBase<U, 2>
{
    typedef RepSecretBase<U, 2> super;

public:
    typedef ReplicatedBase Protocol;

    static ReplicatedSecret constant(const typename super::clear& value,
        int my_num, typename super::mac_key_type, int = -1)
    {
      ReplicatedSecret res;
      if (my_num < 2)
        res[my_num] = value;
      return res;
    }

    ReplicatedSecret() {}
    template <class T>
    ReplicatedSecret(const T& other) : super(other) {}

    void load_clear(int n, const Integer& x);

    BitVec local_mul(const ReplicatedSecret& other) const;

    void reveal(size_t n_bits, Clear& x);
};

class SemiHonestRepPrep;

class SmallRepSecret : public FixedVec<BitVec_<unsigned char>, 2>
{
    typedef FixedVec<BitVec_<unsigned char>, 2> super;
    typedef SmallRepSecret This;

public:
    typedef ReplicatedMC<This> MC;
    typedef BitVec_<unsigned char> open_type;
    typedef open_type clear;
    typedef NoValue mac_key_type;

    static MC* new_mc(mac_key_type)
    {
        return new MC;
    }

    SmallRepSecret()
    {
    }
    template<class T>
    SmallRepSecret(const T& other) :
            super(other)
    {
    }

    This lsb() const
    {
        return *this & 1;
    }
};

class SemiHonestRepSecret : public ReplicatedSecret<SemiHonestRepSecret>
{
    typedef ReplicatedSecret<SemiHonestRepSecret> super;

public:
    typedef Memory<SemiHonestRepSecret> DynamicMemory;

    typedef ReplicatedMC<SemiHonestRepSecret> MC;
    typedef Replicated<SemiHonestRepSecret> Protocol;
    typedef MC MAC_Check;
    typedef SemiHonestRepPrep LivePrep;
    typedef ReplicatedInput<SemiHonestRepSecret> Input;

    typedef SemiHonestRepSecret part_type;
    typedef SmallRepSecret small_type;
    typedef SemiHonestRepSecret whole_type;

    static const bool expensive_triples = false;

    static MC* new_mc(mac_key_type) { return new MC; }

    SemiHonestRepSecret() {}
    template<class T>
    SemiHonestRepSecret(const T& other) : super(other) {}
};

}

#endif /* GC_SHARESECRET_H_ */
