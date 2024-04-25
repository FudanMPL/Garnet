/*
 * MatmulsTuple.h
 *
 */

#ifndef PROCESSOR_MATMULSTUPLE_H_
#define PROCESSOR_MATMULSTUPLE_H_

#include <vector>
using namespace std;

class MatmulsTuple
{
public:
    int dim[3];
    int a, b, c;
    vector<vector<vector<int>>> lengths;

    MatmulsTuple(const vector<int>& args, int start);

    template<class T>
    void pre(const vector<T>& source, vector<T>& S, typename T::Protocol& protocol);
    template<class T>
    void post(vector<T>& S, typename T::Protocol& protocol);

    template<class T>
    void run_matrix(SubProcessor<T>& processor);
};

#endif /* PROCESSOR_MATMULSMTUPLE_H_ */