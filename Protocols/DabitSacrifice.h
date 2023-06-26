/*
 * DabitSacrifice.h
 *
 */

#ifndef PROTOCOLS_DABITSACRIFICE_H_
#define PROTOCOLS_DABITSACRIFICE_H_

template<class T>
class DabitSacrifice
{
    const int S;

public:
    DabitSacrifice();

    int minimum_n_inputs(int n_outputs = 0)
    {
        if (n_outputs < 1)
            n_outputs = OnlineOptions::singleton.batch_size;
        if (T::clear::N_BITS < 0)
            // sacrifice uses S^2 random bits
            n_outputs = max(n_outputs, 10 * S * S);
        return n_outputs + S;
    }

    void sacrifice_without_bit_check(vector<dabit<T>>& dabits,
            vector<dabit<T>>& check_dabits, SubProcessor<T>& proc,
            ThreadQueues* queues = 0);

    void sacrifice_and_check_bits(vector<dabit<T>>& dabits,
            vector<dabit<T>>& check_dabits, SubProcessor<T>& proc,
            ThreadQueues* queues = 0);
};

#endif /* PROTOCOLS_DABITSACRIFICE_H_ */
