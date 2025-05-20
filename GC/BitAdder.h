/*
 * BitAdder.h
 *
 */

#ifndef GC_BITADDER_H_
#define GC_BITADDER_H_

#include <vector>
using namespace std;

class BitAdder
{
public:
    template<class T>
    void add(vector<vector<T>>& res, const vector<vector<vector<T>>>& summands,
            SubProcessor<T>& proc, int length, ThreadQueues* queues = 0,
            int player = -1);

    template<class T>
    void add(vector<vector<T>>& res, const vector<vector<vector<T>>>& summands,
            size_t begin, size_t end, SubProcessor<T>& proc, int length,
            int input_begin = -1, const void* supply = 0);

    template<class T>
    void multi_add(vector<vector<T>>& res, const vector<vector<vector<T>>>& summands,
            size_t begin, size_t end, SubProcessor<T>& proc, int length,
            int input_begin);
};

#endif /* GC_BITADDER_H_ */
