/*
 * MatmulsmTuple.h
 *
 */

#ifndef PROCESSOR_MATMULSMTUPLE_H_
#define PROCESSOR_MATMULSMTUPLE_H_

#include <vector>

using namespace std;

class MatmulsmTuple
{
public:
    int dim[9];
    long a, b;
    int c;
    vector<vector<vector<int>>> lengths;

    MatmulsmTuple(const vector<int>& args, ArithmeticProcessor* Proc, int start);

    template<class T>
    void pre(const vector<T>& source, ArithmeticProcessor* Proc, typename T::Protocol& protocol);
    template<class T>
    void post(vector<T>& S, typename T::Protocol& protocol);

    template<class T>
    void run_matrix(SubProcessor<T>& processor, CheckVector<T>& source);
};

#endif /* PROCESSOR_MATMULSMTUPLE_H_ */