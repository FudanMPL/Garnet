/*
 * VssBeaver.h
 *
 */

#ifndef PROTOCOLS_VSSBEAVER_H_
#define PROTOCOLS_VSSBEAVER_H_

#include <vector>
#include <array>
using namespace std;

#include "Replicated.h"
#include "Processor/Data_Files.h"

template <class T>
class SubProcessor;
template <class T>
class MAC_Check_Base;
class Player;

/**
 * VssBeaver multiplication
 */
template <class T>
class VssBeaver : public ProtocolBase<T>
{
protected:
  vector<T> shares;
  vector<typename T::open_type> opened;
  vector<array<T, 3>> triples;
  vector<int> lengths;
  typename vector<typename T::open_type>::iterator it;
  typename vector<array<T, 3>>::iterator triple;
  Preprocessing<T> *prep;
  typename T::MAC_Check *MC;

public:
  static const bool uses_triples = true;

  Player &P;

  VssBeaver(Player &P) : prep(0), MC(0), P(P) {
    // int public_matrix_row = this->P.num_players(); // n+nd
    // // int public_matrix_col = P.num_players() - ndparties; // n
    // int public_matrix_col = this->P.num_players(); // n+nd

    // public_matrix.resize(public_matrix_row);
    // field_inv.resize(public_matrix_col);

    // for (int i = 0; i < public_matrix_row; i++)
    // {
    //     public_matrix[i].resize(public_matrix_col);
    // }
    // for (int i = 0; i < public_matrix_row; i++)
    // {
    //     int x = 1;
    //     public_matrix[i][0] = 1;
    //     for (int j = 1; j < public_matrix_col; j++)
    //     {
    //         x *= (i + 1);
    //         public_matrix[i][j] = x;
    //     }
    // }

    // vector<vector<int>> selected(public_matrix.begin(), public_matrix.begin() + public_matrix_col);
    // typename T::open_type det = determinant(selected);                   // 行列式
    // typename T::open_type det_inv = det.invert();                        // 行列式的逆
    // vector<vector<typename T::open_type>> adj = adjointMatrix(selected); // 伴随矩阵
    // for (int i = 0; i < public_matrix_col; i++)
    // {
    //     field_inv[i] = adj[0][i] * det_inv; // 逆矩阵的第一行
    // }
  }

  typename T::Protocol branch();

  void init(Preprocessing<T> &prep, typename T::MAC_Check &MC);

  void init_mul();
  void prepare_mul(const T &x, const T &y, int n = -1);
  void exchange();
  T finalize_mul(int n = -1);

  void check();

  void start_exchange();
  void stop_exchange();

  int get_n_relevant_players() { return 1 + T::threshold(P.num_players()); }
};

#endif /* PROTOCOLS_VSSBEAVER_H_ */
