//
// Created by 林国鹏 on 2023/6/2.
//

#ifndef GARNET_TREEINFERENCESERVER_H
#define GARNET_TREEINFERENCESERVER_H

#include <iostream>
#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"

using namespace std;
using namespace seal;

#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "../Tools/int.h"
#include "Tools/TimerWithComm.h"


const size_t poly_modulus_degree = 8192;
const uint64_t plain_modulus = 1024; // 满足十分类
const int scale_for_decimal_part = 100;
const int value_max_threhold = 10;

class EncryptedSample{
public:
  vector<vector<Ciphertext> > features_vector;
  Ciphertext label;
};

class PredictValue{
public:
  Ciphertext judgment;
  Plaintext label;
  bool is_leaf;
  PredictValue* left_value;
  PredictValue* right_value;
  Ciphertext get_final_result();
};

class Node{
public:
  Node(int feature_id, int value, bool is_leaf);
  int feature_id;
  int threhold;
  int label;
  bool is_leaf;
  Node* left_child_node = nullptr;
  Node* right_child_node = nullptr;
  PredictValue* predict(vector<vector<Ciphertext> > &features_vector);
};

class TreeInferenceServer {
public:
  TimerWithComm timer;
  const int playerno = 0;
  const int nplayers = 2;
  int feature_number;
  int tree_number;
  int tree_h;
  int test_sample_number;
  RealTwoPartyPlayer* player;


  vector<Node*> roots;



  TreeInferenceServer();
  void start_networking(ez::ezOptionParser& opt);
  void recv_single_query(vector<vector<Ciphertext> > &features_vector);
  Ciphertext process_single_query(vector<vector<Ciphertext> > &features_vector);
  void expend_index();
  void read_tree_structure();
  void run();
  void read_tree_structure_and_feature_info();
  void recv_keys();
  void send_single_answer(Ciphertext &answer);
  void merge_node(Node* node);
};



#endif //GARNET_TREEINFERENCESERVER_H
