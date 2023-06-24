//
// Created by 林国鹏 on 2023/6/2.
//

#ifndef GARNET_TREEINFERENCECLIENT_H
#define GARNET_TREEINFERENCECLIENT_H

#include <iostream>
#include <vector>
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



class Sample{
public:
  vector<int> features;
  int label;
};


class TreeInferenceClient {
public:
  int feature_number;
  int test_sample_number;
  int tree_number;
  int tree_h;

  vector<Sample>  samples;
  vector<int> max_values;
  TimerWithComm timer;
  const int playerno = 1;
  const int nplayers = 2;
  SEALContext* context;
  KeyGenerator* keygen;
  const SecretKey* secret_key;
  PublicKey* public_key;
  RelinKeys* relin_keys;

  Encryptor* encryptor;
  Evaluator* evaluator;
  Decryptor* decryptor;

  RealTwoPartyPlayer* player;
  TreeInferenceClient();
  void start_networking(ez::ezOptionParser& opt);
  void send_single_query(vector<vector<Ciphertext> >& query);
  int recv_single_answer();
  void read_meta_and_sample();
  void generate_single_query(vector<int> &features,  vector<vector<Ciphertext> >& query);
  void run();
  void send_keys();
};


#endif //GARNET_TREEINFERENCECLIENT_H
