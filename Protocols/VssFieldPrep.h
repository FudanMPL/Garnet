/*
 * VssFieldPrep.h
 *
 */

#ifndef PROTOCOLS_VSSFIELDPREP_H_
#define PROTOCOLS_VSSFIELDPREP_H_

#include "MascotPrep.h"

template <class T>
class VssFieldPrep : public virtual OTPrep<T>, public virtual SemiHonestRingPrep<T>
{
    typedef typename T::open_type open_type;
public:
    VssFieldPrep(SubProcessor<T> *proc, DataPositions &usage) : BufferPrep<T>(usage),
                                                           BitPrep<T>(proc, usage),
                                                           OTPrep<T>(proc, usage),
                                                           RingPrep<T>(proc, usage),
                                                           SemiHonestRingPrep<T>(proc, usage)
    {
        this->params.set_passive();
    }

    Integer determinant(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        if (n == 2)
        {
            Integer det = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
            return det;
        }
        Integer det = 0;
        bool sign = true;
        for (int i = 0; i < n; i++)
        {
            vector<vector<int>> submatrix(n - 1, vector<int>(n - 1));
            for (int j = 1; j < n; j++)
            {
                int col = 0;
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        submatrix[j - 1][col] = matrix[j][k];
                        col++;
                    }
                }
            }
            if (sign == true)
                det = det + (determinant(submatrix) * matrix[0][i]);
            else
                det = det - (determinant(submatrix) * matrix[0][i]);
            sign = !sign;
        }
        return det;
    }

    vector<vector<typename T::open_type>> adjointMatrix(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        vector<vector<typename T::open_type>> adj(n, vector<typename T::open_type>(n));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                vector<vector<int>> submatrix(n - 1, vector<int>(n - 1));
                int subi = 0, subj = 0;
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        subj = 0;
                        for (int l = 0; l < n; l++)
                        {
                            if (l != j)
                            {
                                submatrix[subi][subj] = matrix[k][l];
                                subj++;
                            }
                        }
                        subi++;
                    }
                }
                int sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj[j][i] = Integer(sign) * determinant(submatrix);
            }
        }
        return adj;
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
        return res; // 返回 份额之和
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

#endif /* PROTOCOLS_VssFieldPrep_H_ */