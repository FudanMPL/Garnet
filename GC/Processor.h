/*
 * Processor.h
 *
 */

#ifndef GC_PROCESSOR_H_
#define GC_PROCESSOR_H_

#include <vector>
using namespace std;

#include "GC/Clear.h"
#include "GC/Machine.h"
#include "GC/RuntimeBranching.h"

#include "Math/Integer.h"
#include "Processor/ProcessorBase.h"
#include "Processor/Instruction.h"

namespace GC
{

template <class T>
class Processor : public ::ProcessorBase, public GC::RuntimeBranching
{
public:
    static int check_args(const vector<int>& args, int n);

    template<class U>
    static void check_input(const U& in, const int* params);

    Machine<T>* machine;
    Memories<T>& memories;

    unsigned int PC;
    unsigned int time;

    // rough measure for the memory usage
    size_t complexity;

    Memory<T> S;
    Memory<Clear> C;
    Memory<Integer> I;

    Timer xor_timer;

    typename T::out_type out;

    Processor(Machine<T>& machine);
    Processor(Memories<T>& memories, Machine<T>* machine = 0);
    ~Processor();

    template<class U>
    void reset(const U& program, int arg);
    template<class U>
    void reset(const U& program);

    long long get_input(const int* params, bool interactive = false);
    template<class U>
    U get_long_input(const int* params, ProcessorBase& input_proc,
            bool interactive = false);

    void bitcoms(T& x, const vector<int>& regs) { x.bitcom(S, regs); }
    void bitdecs(const vector<int>& regs, const T& x) { x.bitdec(S, regs); }
    void bitdecc(const vector<int>& regs, const Clear& x);
    void bitdecint(const vector<int>& regs, const Integer& x);

    void random_bit(T &x) { x.random_bit(); }

    template<class U>
    void load_dynamic_direct(const vector<int>& args, U& dynamic_memory);
    template<class U>
    void store_dynamic_direct(const vector<int>& args, U& dynamic_memory);
    template<class U>
    void load_dynamic_indirect(const vector<int>& args, U& dynamic_memory);
    template<class U>
    void store_dynamic_indirect(const vector<int>& args, U& dynamic_memory);
    template<class U>
    void store_clear_in_dynamic(const vector<int>& args, U& dynamic_memory);

    template<class U>
    void mem_op(int n, Memory<U>& dest, const Memory<U>& source,
            Integer dest_address, Integer source_address);

    void xors(const vector<int>& args);
    void xors(const vector<int>& args, size_t start, size_t end);
    void xorc(const ::BaseInstruction& instruction);
    void nots(const ::BaseInstruction& instruction);
    void notcb(const ::BaseInstruction& instruction);
    void andm(const ::BaseInstruction& instruction);
    void and_(const vector<int>& args, bool repeat);
    void andrs(const vector<int>& args) { and_(args, true); }
    void ands(const vector<int>& args) { and_(args, false); }
    void andrsvec(const vector<int>& args);

    void input(const vector<int>& args);
    void inputb(typename T::Input& input, ProcessorBase& input_processor,
            const vector<int>& args, int my_num);
    void inputbvec(typename T::Input& input, ProcessorBase& input_processor,
            const vector<int>& args, PlayerBase& P);

    void reveal(const vector<int>& args);

    template<int = 0>
    void convcbit2s(const BaseInstruction& instruction);

    void print_reg(int reg, int n, int size);
    void print_reg_plain(Clear& value);
    void print_reg_signed(unsigned n_bits, Integer value);
    void print_chr(int n);
    void print_str(int n);
    void print_float(const vector<int>& args);
    void print_float_prec(int n);
};

template <class T>
inline int GC::Processor<T>::check_args(const vector<int>& args, int n)
{
    if (args.size() % n != 0)
        throw runtime_error("invalid number of arguments");
    int total = 0;
    for (auto it = args.begin(); it < args.end(); it += n)
    {
        total += *it;
    }
    return total;
}

} /* namespace GC */

#endif /* GC_PROCESSOR_H_ */
