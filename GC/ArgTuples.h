/*
 * ArgTuples.h
 *
 */

#ifndef GC_ARGTUPLES_H_
#define GC_ARGTUPLES_H_

#include <vector>
using namespace std;

template <class T>
class ArgIter
{
    vector<int>::const_iterator it;
    vector<int>::const_iterator end;

public:
    ArgIter(const vector<int>::const_iterator it,
            const vector<int>::const_iterator end) :
                it(it), end(end)
    {
    }

    T operator*()
    {
        return it;
    }

    ArgIter<T> operator++()
    {
        auto res = it;
        it += T(res).n;
        if (it > end)
            throw runtime_error("wrong number of args");
        return {res, end};
    }

    bool operator!=(const ArgIter<T>& other)
    {
        return it != other.it;
    }
};

template <class T>
class ArgList
{
    const vector<int>& args;

public:
    ArgList(const vector<int>& args) :
            args(args)
    {
    }

    ArgIter<T> begin()
    {
        return {args.begin(), args.end()};
    }

    ArgIter<T> end()
    {
        return {args.end(), args.end()};
    }
};

class InputArgs
{
public:
    static const int n = 4;

    int from;
    int& n_bits;
    int& n_shift;
    int params[2];
    int dest;

    InputArgs(vector<int>::const_iterator it) : n_bits(params[0]), n_shift(params[1])
    {
        from = *it++;
        n_bits = *it++;
        n_shift = *it++;
        dest = *it++;
    }
};

template<class T>
class InputArgListBase : public ArgList<T>
{
public:
    InputArgListBase(const vector<int>& args) :
            ArgList<T>(args)
    {
    }

    int n_inputs_from(int from)
    {
        int res = 0;
        for (auto x : *this)
            res += x.from == from;
        return res;
    }

    int n_input_bits()
    {
        int res = 0;
        for (auto x : *this)
            res += x.n_bits;
        return res;
    }

    int n_interactive_inputs_from_me(int my_num);
};

class InputArgList : public InputArgListBase<InputArgs>
{
public:
    InputArgList(const vector<int>& args) :
            InputArgListBase<InputArgs>(args)
    {
    }
};

class InputVecArgs
{
public:
    int from;
    int n;
    int& n_bits;
    int& n_shift;
    int params[2];
    vector<int> dest;

    InputVecArgs(vector<int>::const_iterator it) : n_bits(params[0]), n_shift(params[1])
    {
        n = *it++;
        n_bits = n - 3;
        n_shift = *it++;
        from = *it++;
        dest.resize(n);
        for (int i = 0; i < n_bits; i++)
            dest[i] = *it++;
    }
};

class InputVecArgList : public InputArgListBase<InputVecArgs>
{
public:
    InputVecArgList(const vector<int>& args) :
            InputArgListBase<InputVecArgs>(args)
    {
    }
};

#endif /* GC_ARGTUPLES_H_ */
