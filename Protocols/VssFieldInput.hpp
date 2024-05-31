/*
 * VssFieldInput.cpp
 *
 */

#ifndef PROTOCOLS_VSSFIELDINPUT_HPP_
#define PROTOCOLS_VSSFIELDINPUT_HPP_

#include "VssFieldInput.h"


// 求矩阵的行列式
template <class T>
typename T::open_type determinant(vector<vector<typename T::open_type>> &matrix)
{
    int n = matrix.size();
    if (n == 2)
    {
        typename T::open_type det = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] );
        // if (det < 0)
        // {
        //     det += P;
        // }
        // 需要确认 * + - 等运算的实现细节，是否取模，是否+P
        return det;
    }
    typename T::open_type det = 0;
    typename T::open_type sign = 1;
    for (int i = 0; i < n; i++)
    {
        vector<vector<typename T::open_type>> submatrix(n - 1, vector<typename T::open_type>(n - 1));
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
        det += ((sign * matrix[0][i]) * determinant(submatrix));
        // sign = P - sign;
        sign = -sign;
    }
    // if (det < 0)
    // {
    //     det += P;
    // }
    return det;
}

//求矩阵的伴随矩阵
template <class T>
std::vector<std::vector<typename T::open_type>> adjointMatrix(vector<vector<typename T::open_type>> &matrix)
{
    int n = matrix.size();
    vector<vector<typename T::open_type>> adj(n, vector<typename T::open_type>(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            vector<vector<typename T::open_type>> submatrix(n - 1, vector<typename T::open_type>(n - 1));
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
            // typename T::open_type sign = ((i + j) % 2 == 0) ? 1 : P - 1;
            typename T::open_type sign = ((i + j) % 2 == 0) ? 1 : -1;
            adj[j][i] = sign * determinant(submatrix);
        }
    }
    return adj;
}

// 求a的b次方
typename T::open_type power_field(typename T::open_type a, int b)
{
    typename T::open_type res = 1;
    while (b)
    {
        if (b & 1)
        {
            res *= a;
        }
        a *= a;
        b >>= 1;
    }
    return res;
}

// 求逆元
typename T::open_type inverse(typename T::open_type a)
{
    // return power(a, P - 2);
    return power_field(a, -1); //负数的表示
}

template <class T>
VssFieldInput<T>::VssFieldInput(SubProcessor<T> *proc, Player &P) : SemiInput<T>(proc, P), P(P)
{
    int public_matrix_row = P.num_players(); // n+nd
    int public_matrix_col = P.num_players() - ndparties; // n
    P.public_matrix.resize(public_matrix_row);
    inv.resize(public_matrix_col);
    for (int i = 0; i < public_matrix_row; i++)
    {
        P.public_matrix[i].resize(public_matrix_col);
    }
    os.resize(2); // 是什么，socket发送
    os[0].resize(P.public_matrix[0].size());
    os[1].resize(P.public_matrix[0].size());
    expect.resize(P.public_matrix[0].size()); // 是什么
    for (int i = 0; i < public_matrix_row; i++)
    {
        typename T::open_type x = 1;
        for (int j = 0; j < public_matrix_col; j++){
            x *= (i + 1);
            P.public_matrix[i][j] = x;
        }
    }
    // 求前n行的行列式
    vector<vector<typename T::open_type>> selected(P.public_matrix.begin(), P.public_matrix.begin() + public_matrix_col);
    typename T::open_type det = determinant(selected); // 行列式
    vector<vector<typename T::open_type>> adj = adjointMatrix(selected); // 伴随矩阵
    typename T::open_type det_inv = inverse(det); // 行列式的逆
    vector<vector<typename T::open_type>> selected_inv(public_matrix_col,vector<typename T::open_type>(public_matrix_col));
    for (int i = 0; i < public_matrix_col; i++)
    {
        for (int j = 0; j < public_matrix_col; j++)
        {
            selected_inv[i][j] = adj[i][j] * det_inv; // 伴随矩阵 * 行列式的逆 = 矩阵的逆
        }
    }
    for (int i = 0; i < public_matrix_col; i++)
    {
        inv[i] = selected_inv[0][i];
    } 
    // for test，输出生成的范德蒙矩阵P.public_matrix及其逆矩阵inv
    cout << "范德蒙矩阵测试" << endl;
    for (int i = 0; i < public_matrix_col; i++)
    {
        for (int j = 0; j < public_matrix_col; j++)
        {
            cout << P.public_matrix[i][j] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < public_matrix_col; i++)
    {
        cout << inv[i] << " ";
    }

    this->reset_all(P);
}

template <class T>
void VssFieldInput<T>::reset(int player)
{
    if (player == P.my_num())
    {
        this->shares.clear();
        os.resize(2);
        for (int i = 0; i < 2; i++)
        {
            os[i].resize(P.public_matrix[0].size());
            for (auto &o : os[i])
                o.reset_write_head();
        }
    }
    expect[player] = false;
}

template <class T>
void VssFieldInput<T>::add_mine(const typename T::clear &input, int) // 计算秘密份额
{ 
    auto &P = this->P;
    vector<typename T::open_type> v(P.public_matrix[0].size());
    vector<T> secrets(P.public_matrix.size());
    PRNG G;
    v[0] = input;
    for (int i = 1; i < P.public_matrix[0].size(); i++)
    {
        v[i] = G.get<typename T::open_type>();
    }
    for (int i = 0; i < P.public_matrix.size(); i++)
    {
        typename T::open_type sum = 0;
        for (int j = 0; j < P.public_matrix[0].size(); j++)
        {
            sum += v[j] * P.public_matrix[i][j];
        }
        secrets[i] = sum;
    }
    this->shares.push_back(secrets[P.my_num()]);
    for (int i = 0; i < P.num_players(); i++)
    {
        if (i != P.my_num())
        {
            secrets[i].pack(os[0][i]);
        }
    }
    // typename T::open_type sum;
    // std::vector<typename T::open_type> shares(P.num_players());
    // for (int i = 0; i < P.num_players(); i++)
    // {
    //     if (i != P.my_num())
    //     {
    //         sum += this->send_prngs[i].template get<typename T::open_type>() * P.inv[i];
    //         // sum += this->send_prngs[i].template get<typename T::open_type>();
    //     }
    // }
    // cout << sum <<endl;
    // bigint temp = input - sum;
    // stringstream ss;
    // ss << temp;
    // long value = stol(ss.str()) / P.inv[P.my_num()];
    // cout << value << endl;
    // this->shares.push_back(value);
}

template <class T>
void VssFieldInput<T>::add_other(int player, int)
{
    expect[player] = true;
}

template <class T>
void VssFieldInput<T>::exchange()
{
    if (!os[0][(P.my_num() + 1) % P.num_players()].empty()) // 如果不为空
    {
        for (int i = 0; i < P.num_players(); i++)
        {
            if (i != P.my_num())
                P.send_to(i, os[0][i]); // 发送数据(秘密份额)
        }
        for (int i = 0; i < P.num_players(); i++)
        {
            if (expect[i]) // 从expect[i]的参与者处接收数据
                P.receive_player(i, os[1][i]);
        }
        
    }
    else // 如果为空，无需发送数据
    {
        for (int i = 0; i < P.num_players(); i++)
        {
            if (expect[i])
                P.receive_player(i, os[1][i]);
        }
    }
}

template <class T>
void VssFieldInput<T>::finalize_other(int player, T &target, octetStream &,
                                 int)
                          // 从其他参与者那里接收的数据存到target中       
{
    target = os[1][player].template get<T>();
}

template <class T>
T VssFieldInput<T>::finalize_mine() // 获取并返回shares的下一个元素？
{
    return this->shares.next();
}

#endif  // PROTOCOLS_VSSFIELDINPUT_HPP_
