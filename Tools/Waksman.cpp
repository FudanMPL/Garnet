/*
 * Waksman.cpp
 *
 */

#include "Waksman.h"

#include <algorithm>
#include <assert.h>
#include <math.h>

template<class T>
void append(vector<T>& x, const vector<T>& y)
{
    x.insert(x.end(), y.begin(), y.end());
}

vector<vector<bool> > Waksman::configure(const vector<int>& perm)
{
    int n = perm.size();
    assert(n > 1);

    if (n == 2)
        return {{perm[0] == 1, perm[0] == 1}};

    vector<bool> I(n / 2);
    vector<char> O(n / 2, -1);
    vector<int>  p0(n / 2, -1), p1(n / 2, -1), inv_perm(n);

    for (int i = 0; i < n; i++)
        inv_perm[perm[i]] = i;

    while (true)
    {
        auto it = find(O.begin(), O.end(), -1);
        if (it == O.end())
            break;
        int j = 2 * (it - O.begin());
        O.at(j / 2) = 0;
        int j0 = j;

        while (true)
        {
            int i = inv_perm.at(j);
            p0.at(i / 2) = j / 2;
            I.at(i / 2) = i % 2;
            O.at(j / 2) = j % 2;
            if (i % 2 == 1)
                i--;
            else
                i++;
            j = perm.at(i);
            if (j % 2 == 1)
                j--;
            else
                j++;
            p1.at(i / 2) = perm.at(i) / 2;
            if (j == j0)
                break;
        }

        if ((find(p1.begin(), p1.end(), -1) == p1.end())
                and (find(p0.begin(), p0.end(), -1) == p0.end()))
            break;
    }

    auto p0_config = configure(p0);
    auto p1_config = configure(p1);

    vector<vector<bool>> res;
    res.push_back(I);
    for (auto& x : O)
        res.back().push_back(x);

    assert(p0_config.size() == p1_config.size());

    for (size_t i = 0; i < p0_config.size(); i++)
    {
        res.push_back(p0_config.at(i));
        append(res.back(), p1_config.at(i));
    }

    assert(res.size() == Waksman(perm.size()).n_rounds());
    return res;
}

Waksman::Waksman(int n_elements) :
        n_elements(n_elements), nr(log2(n_elements))
{
    assert(n_elements == (1 << nr));
}
