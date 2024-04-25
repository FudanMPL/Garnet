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


template<class T>
class Fss : public ReplicatedBase, public ProtocolBase<T>
{
    array<octetStream, 2> os;
    PointerVector<typename T::clear> add_shares;
    typename T::clear dotprod_share;
    Preprocessing<T>* prep;
    typename T::LivePrep* fss3prep;
    typename T::MAC_Check* MC;

    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc, true_type);
    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc, false_type);


public:
    static const bool uses_fss_cw = true;
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

    void init_fss_cmp_prep(SubProcessor<T> &proc);
    void init_fss_conv_relu_prep(SubProcessor<T> &proc, int float_bits, int case_num=1);
    //initialize preprocessing for fss preprocess
    void init(Preprocessing<T>& prep, typename T::MAC_Check& MC);
    void init_mul();
    void prepare_mul(const T &x, const T &y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);

    void fss_cmp(SubProcessor<T> &proc, const Instruction &instruction);
    void prepare_reshare(const typename T::clear &share, int n = -1);

    void init_dotprod();
    void prepare_dotprod(const T &x, const T &y);
    void next_dotprod();
    T finalize_dotprod(int length);

    //multiplication without truncation and reshare
    void init_dotprod_without_trunc();
    void prepare_dotprod_without_trunc(const T &x, const T &y);
    void next_dotprod_without_trunc();
    void exchange_without_trunc(){};
    T finalize_dotprod_without_trunc(int length);

    template <class U>
    void trunc_pr(const vector<int> &regs, int size, U &proc);

    T get_random();
    void randoms(T &res, int n_bits);

    void start_exchange();
    void stop_exchange();

    template <class U>
    void change_domain(const vector<int> &reg, U &proc);

    //new added function    
    void distributed_comparison_function(SubProcessor<T> &proc, const Instruction &instruction, int lambda);

    void distributed_comparison_function_gpu(SubProcessor<T> &proc, const Instruction &instruction, int lambda);

    //new added generate function
    void generate();

    //new added evaluate function
    bigint evaluate(typename T::open_type x, int lambda, int result_length, int drop_least_bits = 0);
    bigint evaluate_conv_relu(typename T::open_type x, int n, int result_length);
    //Instructions for RFss3
    void conv2d_rfss3s(SubProcessor<T> &proc, const Instruction& instruction);
    void trunc_relu_rfss3s(SubProcessor<T> &proc, const Instruction& instruction);
};


#endif /* PROTOCOLS_Fss_H_ */