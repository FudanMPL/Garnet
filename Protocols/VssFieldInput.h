

#ifndef PROTOCOLS_VSSFIELDINPUT_H_
#define PROTOCOLS_VSSFIELDINPUT_H_

#include "SemiInput.h"
#include "ReplicatedInput.h"
#include "Processor/Input.h"
#include "Vss.h"
#include "Machines/vss-field-party.h"

template <class T>
class VssFieldMC;

/**
 * Vector space secret sharing over field input protocol
 */
template <class T>
class VssFieldInput : public SemiInput<T>
{
    Player &P;
    // friend class Vss<T>;
    vector<vector<octetStream>> os;
    vector<bool> expect;
    // int npparties;   // the number of privileged parties
    // int naparties;   // the number of assistant parties
    int ndparties; // the number of assistant parties allowed to drop out
    vector<vector<typename T::open_type>> public_matrix;

public:
    VssFieldInput(SubProcessor<T> &proc, VssFieldMC<T> &) : VssFieldInput(&proc, proc.P)
    {
        cout << "我在VssFieldInput第一个构造函数" << endl;
        int public_matrix_row = P.num_players();             // n+nd
        int public_matrix_col = P.num_players() - ndparties; // n
        public_matrix.resize(public_matrix_row);
        P.public_matrix.resize(public_matrix_row);
        for (int i = 0; i < public_matrix_row; i++)
        {
            public_matrix[i].resize(public_matrix_col);
            P.public_matrix[i].resize(public_matrix_col);
        }
        for (int i = 0; i < public_matrix_row; i++)
        {
            int x = 1;
            public_matrix[i][0] = 1;
            P.public_matrix[i][0] = 1;
            for (int j = 1; j < public_matrix_col; j++)
            {
                x *= (i + 1);
                public_matrix[i][j] = x;
                P.public_matrix[i][j] = x;
            }
        }
    }

    VssFieldInput(SubProcessor<T> *proc, Player &P);

    VssFieldInput(typename T::MAC_Check &MC, Preprocessing<T> &prep, Player &P) : VssFieldInput(0, P)
    {
        (void)MC, (void)prep;
    }

    void reset(int player);
    void add_mine(const typename T::clear &input, int n_bits = -1);
    void add_other(int player, int n_bits = -1);
    void exchange();
    void finalize_other(int player, T &target, octetStream &o, int n_bits = -1);
    T finalize_mine();

    Integer determinant(vector<vector<int>> &matrix);                                 // 行列式
    vector<vector<typename T::open_type>> adjointMatrix(vector<vector<int>> &matrix); // 伴随矩阵
};

#endif /* PROTOCOLS_VSSFIELDINPUT_H_ */
