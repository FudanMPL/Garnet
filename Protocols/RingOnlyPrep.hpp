/*
 * RingOnlyPrep.cpp
 *
 */

#include "RingOnlyPrep.h"


template<class T>
void RingOnlyPrep<T>::buffer_dabits_from_bits_without_check(
        vector<dabit<T> >& dabits, int buffer_size, ThreadQueues*)
{
    vector<dabit<T>> new_dabits;
    assert(this->proc != 0);
    auto& party = GC::ShareThread<typename T::bit_type>::s();
    typedef typename T::bit_type::part_type BT;
    SubProcessor<BT> bit_proc(party.MC->get_part_MC(),
            this->proc->bit_prep, this->proc->P);
    typename T::bit_type::part_type::Input input(bit_proc);
    input.reset_all(this->proc->P);
    for (int i = 0; i < buffer_size; i++)
    {
        T bit;
        this->get_one(DATA_BIT, bit);
        new_dabits.push_back({bit, {}});
        input.add_from_all(bit.get_share().get_bit(0));
    }
    input.exchange();
    for (size_t i = 0; i < new_dabits.size(); i++)
        for (int j = 0; j < this->proc->P.num_players(); j++)
            new_dabits[i].second += input.finalize(j);
    dabits.insert(dabits.end(), new_dabits.begin(), new_dabits.end());
}
