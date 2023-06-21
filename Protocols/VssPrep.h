/*
 * VssPrep.h
 *
 */

#ifndef PROTOCOLS_VSSPREP_H_
#define PROTOCOLS_VSSPREP_H_

#include "MascotPrep.h"

template <class T>
class VssPrep : public virtual OTPrep<T>, public virtual SemiHonestRingPrep<T>
{
    typedef typename T::open_type open_type;
public:
    VssPrep(SubProcessor<T> *proc, DataPositions &usage) : BufferPrep<T>(usage),
                                                           BitPrep<T>(proc, usage),
                                                           OTPrep<T>(proc, usage),
                                                           RingPrep<T>(proc, usage),
                                                           SemiHonestRingPrep<T>(proc, usage)
    {
        this->params.set_passive();
    }
    open_type toVSSTriples(open_type X)
    {
        octetStream os, oc;
        auto&P = this->proc->P;
        int n = P.num_players();
        vector<open_type> my_share(n, 0);
        vector<open_type> Vss_X(n, 0);
        vector<open_type> S(n);
        open_type res = 0;
        SeededPRNG G;

        S[0] = X;
        for (int i = 1; i < n; i++)
        {
            S[i] = G.get<open_type>();
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Vss_X[i] += S[j] * P.public_matrix[i][j];
            }
        }
        for (int k = 0; k < n; k++)
        {
            os.reset_write_head();
            oc.reset_read_head();

            if (P.my_num() != k)
                Vss_X[k].pack(os);
            else
                my_share[k] = Vss_X[k];
            if (P.my_num() != k)
            {
                P.send_to(k, os);
                P.receive_player(k, oc);
                my_share[k] = oc.get<open_type>();
            }

        }
        for (int k = 0; k < n; k++)
        {
            res += my_share[k];
        }
        return res;
    }
    void buffer_triples()
    {
        assert(this->triple_generator);
        this->triple_generator->generatePlainTriples();
        for (auto &x : this->triple_generator->plainTriples)
        {
            x[0] = toVSSTriples(x[0]);
            x[1] = toVSSTriples(x[1]);
            x[2] = toVSSTriples(x[2]);
            this->triples.push_back({{x[0], x[1], x[2]}});
        }
        this->triple_generator->unlock();
    }
};

#endif /* PROTOCOLS_VSSPREP_H_ */