/*
 * Waksman.h
 *
 */

#ifndef TOOLS_WAKSMAN_H_
#define TOOLS_WAKSMAN_H_

#include <vector>
using namespace std;

class Waksman
{
    int n_elements;
    int nr;

public:
    static vector<vector<bool>> configure(const vector<int>& perm);

    Waksman(int n_elements);

    size_t n_rounds() const
    {
        return nr;
    }

    bool matters(int i, int j) const
    {
        int block = n_elements >> i;
        return (block == 2) or j % block != block / 2;
    }

    bool is_double(int i, int j)
    {
        return (i == (nr - 1) and j % 2 == 1);
    }

    size_t n_bits() const
    {
        return nr * n_elements - n_elements + 1;
    }
};

#endif /* TOOLS_WAKSMAN_H_ */
