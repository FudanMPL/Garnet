/*
 * Rep3Shuffler.h
 *
 */

#ifndef PROTOCOLS_REP3SHUFFLER_H_
#define PROTOCOLS_REP3SHUFFLER_H_

template<class T>
class Rep3Shuffler
{
    SubProcessor<T>& proc;

    vector<array<vector<int>, 2>> shuffles;

public:
    Rep3Shuffler(vector<T>& a, size_t n, int unit_size, size_t output_base,
            size_t input_base, SubProcessor<T>& proc);

    Rep3Shuffler(SubProcessor<T>& proc);

    int generate(int n_shuffle);

    void apply(vector<T>& a, size_t n, int unit_size, size_t output_base,
            size_t input_base, int handle, bool reverse);

    void inverse_permutation(vector<T>& stack, size_t n, size_t output_base,
            size_t input_base);

    void del(int handle);
};

#endif /* PROTOCOLS_REP3SHUFFLER_H_ */
