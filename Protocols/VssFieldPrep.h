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

        vector<vector<open_type>> public_matrix;
        int public_matrix_row = P.num_players(); // n+nd
        int public_matrix_col = P.num_players(); // n+nd, 为了测试，暂时设为n+nd
        public_matrix.resize(public_matrix_row);
        for (int i = 0; i < public_matrix_row; i++)
        {
            public_matrix[i].resize(public_matrix_col);
        }
        for (int i = 0; i < public_matrix_row; i++)
        {
            int x = 1;
            public_matrix[i][0] = 1;
            for (int j = 1; j < public_matrix_col; j++)
            {
                x *= (i + 1);
                public_matrix[i][j] = x;
            }
        }

        // int array[4][3] = {{1, 0, 1},
        //                     {2, 2, -3},
        //                     {3, 3, -4},
        //                     {1, 1, -1}};

        // for (int i = 0; i < public_matrix_row; i++)
        // {
        //     for (int j = 0; j < public_matrix_col; j++)
        //     {
        //         public_matrix[i][j] = array[i][j];
        //     }
        // }

        S[0] = X;
        for (int i = 1; i < n; i++)
        {
            // S[i] = G.get<open_type>();  // for test, 记得改回来
            S[i] = i;
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Vss_X[i] += S[j] * public_matrix[i][j]; // 是share份额
            }
        }
        // cout << "X" << X << endl;
        // for (int i = 0; i < n; i++)
        // {
        //     cout << "Vss_X[" << i << "]: " << Vss_X[i] << endl;
        // }

        for (int k = 0; k < n; k++) // share份额赋值给my_share
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
            // cout << endl;
            // cout << "x[0]: " << x[0] << endl;
            // cout << "x[1]: " << x[1] << endl;
            // cout << "x[2]: " << x[2] << endl;
            // x123是三元组share后本player持有的份额，本人的x1+其他人的x1=x1
            x[0] = toVSSTriples(x[0]);
            x[1] = toVSSTriples(x[1]);
            x[2] = toVSSTriples(x[2]);
            // cout << "x[0]: " << x[0] << endl;
            // cout << "x[1]: " << x[1] << endl;
            // cout << "x[2]: " << x[2] << endl;
            this->triples.push_back({{x[0], x[1], x[2]}});
        }
        this->triple_generator->unlock();
    }
};

#endif /* PROTOCOLS_VssFieldPrep_H_ */