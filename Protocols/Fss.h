/*
 * FSS.h
 *
 */

#ifndef PROTOCOLS_Fss_H_
#define PROTOCOLS_Fss_H_

#include <assert.h>
#include <vector>
#include <array>
#include <map>
#include <string>
using namespace std;

#include "Protocols/Fss3Prep.h"
#include "Tools/octetStream.h"
#include "Tools/random.h"
#include "Tools/PointerVector.h"
#include "Networking/Player.h"
#include <typeinfo>
#include "Protocols/Replicated.h"
#define GEN 2
#define EVAL_1 0
#define EVAL_2 1

template <class T>
class Fss : public ReplicatedBase, public ProtocolBase<T>
{
    array<octetStream, 2> os;
    PointerVector<typename T::clear> add_shares;
    typename T::clear dotprod_share;
    Preprocessing<T> *prep;
    Fss3Prep<T> *fss3prep;
    typename T::MAC_Check *MC;

    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc, true_type);
    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc, false_type);

public:
    static const bool uses_triples = false;

    typedef Rep3Shuffler<T> Shuffler;

    Fss(Player &P);
    Fss(const ReplicatedBase &other);

    static void assign(T &share, const typename T::clear &value, int my_num)
    {
        assert(T::vector_length == 2);
        share.assign_zero();
        if (my_num < 2)
            share[my_num] = value;
    }

    // initialize preprocessing for fss preprocess
    void init(Preprocessing<T> &prep, typename T::MAC_Check &MC);
    void init_mul();
    void prepare_mul(const T &x, const T &y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);
    void cisc(SubProcessor<T> &proc, const Instruction &instruction);
    void prepare_reshare(const typename T::clear &share, int n = -1);

    void init_dotprod();
    void prepare_dotprod(const T &x, const T &y);
    void next_dotprod();
    T finalize_dotprod(int length);

    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc);

    T get_random();
    void randoms(T &res, int n_bits);

    void start_exchange();
    void stop_exchange();

    template <class U>
    void psi(const vector<typename T::clear> &source, const Instruction &instruction, U &proc)
    {
        throw not_implemented();
    }

    template <class U>
    void psi_align(const vector<typename T::clear> &source, const Instruction &instruction, U &proc)
    {
        throw not_implemented();
    }

    template <class U>
    void change_domain(const vector<int> &reg, U &proc);

    // new added function
    void distributed_comparison_function(SubProcessor<T> &processor, const Instruction &instruction, int lambda);

    void Muliti_Interval_Containment(SubProcessor<T> &processor, const Instruction &instruction, int lambda);

    // new added generate function
    void generate();

    // new added evaluate function
    bigint evaluate(typename T::clear x, int lambda);
};

#endif /* PROTOCOLS_Fss_H_ */