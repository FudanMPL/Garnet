//
// Created by 林国鹏 on 2023/6/2.
//
#include <iostream>
#include <fstream>


#include "TreeInferenceServer.h"
#include <cmath>

RelinKeys* relin_keys;
PublicKey* public_key;
Evaluator* evaluator;
Encryptor* encryptor;
SEALContext* context;
vector<int> max_values;


// only for test
Decryptor* decryptor_for_test;

Ciphertext PredictValue::get_final_result() {
  Ciphertext ans;
  const std::uint64_t temp_1 = 1;
  Plaintext one(seal::util::uint_to_hex_string(&temp_1, std::size_t(1)));
  Ciphertext left = this->judgment;
  Ciphertext right;
  evaluator->negate(left, right);
  evaluator->add_plain_inplace(right, one);
  if (left_value->is_leaf){
    evaluator->multiply_plain_inplace(left, left_value->label);
  }
  else{
    evaluator->multiply_inplace(left, left_value->get_final_result());
  }
  if (right_value->is_leaf){
    evaluator->multiply_plain_inplace(right, right_value->label);
  } else{
    evaluator->multiply_inplace(right, right_value->get_final_result());
  }
  evaluator->add(left, right, ans);
  return ans;
}

Node::Node(int feature_id, int value, bool is_leaf) {
  this->feature_id = feature_id;
  this->is_leaf = is_leaf;
  if (is_leaf)
    this->label = value;
  else
    this->threhold = value;
}



PredictValue *Node::predict(vector<vector<Ciphertext> > &features_vector) {
  PredictValue* predict_value = new PredictValue();
  if (!is_leaf){
    if (this->left_child_node != nullptr)
      predict_value->left_value = this->left_child_node->predict(features_vector);
    if (this->right_child_node != nullptr)
      predict_value->right_value = this->right_child_node->predict(features_vector);

    if (max_values[feature_id] < value_max_threhold){
      vector<Ciphertext> feature_prefix_sum = features_vector[feature_id];
      predict_value->judgment = feature_prefix_sum[threhold];
      predict_value->is_leaf = false;
    } else{
      // if the max_value is more than the value_max_threhold,
      // feature_row_and_col contains a row vector and a col vector, each of which contains only one '1'
      vector<Ciphertext> feature_row_and_col = features_vector[feature_id];
      int max_value = max_values[feature_id];
      int row = ceil(sqrt(max_value));
      int col = ceil(max_value / row);
      int row_index = threhold / col;
      int begin_value = row_index * col;
//      for (int i = 0; i < (int ) feature_row_and_col.size(); i++){
//        Plaintext tmp;
//        decryptor_for_test->decrypt(feature_row_and_col[i],  tmp);
//        cout << "feature_row_and_col  " << i << " = " <<   tmp.to_string() << endl;
//      }
      Ciphertext encrypted_row = feature_row_and_col[0];
      for (int i = begin_value +1; i < threhold; i++){
        evaluator->add_inplace(encrypted_row, feature_row_and_col[i - begin_value]);
      }
      const std::uint64_t temp_1 = 1;
      Plaintext one(seal::util::uint_to_hex_string(&temp_1, std::size_t(1)));
      const std::uint64_t temp_0 = 0;
      Plaintext zero(seal::util::uint_to_hex_string(&temp_0, std::size_t(1)));
      Ciphertext judgment;
      encryptor->encrypt(zero, judgment);
      for (int i = 0; i < row_index; i++){
        evaluator->add_inplace(judgment, feature_row_and_col[col+i]);
      }
      Ciphertext temp;
      evaluator->multiply(feature_row_and_col[col+row_index], encrypted_row, temp);
      evaluator->relinearize_inplace(temp, *relin_keys);
      evaluator->add_inplace(judgment, temp);
      predict_value->judgment = judgment;
      predict_value->is_leaf = false;
//      Plaintext tmp_jud;
//      decryptor_for_test->decrypt(predict_value->judgment,  tmp_jud);
//      cout << "judgment = " <<  tmp_jud.to_string() << endl;
    }
  } else{
    uint64_t temp = this->label;
    Plaintext value(seal::util::uint_to_hex_string( &temp, std::size_t(1)));
    predict_value->label = value;
    predict_value->is_leaf = true;
  }
  return predict_value;
}


TreeInferenceServer::TreeInferenceServer() {
  EncryptionParameters parms(scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  parms.set_plain_modulus(plain_modulus);
  context = new SEALContext(parms);
  evaluator = new Evaluator(*context);
}


void TreeInferenceServer::start_networking(ez::ezOptionParser& opt) {
  string hostname, ipFileName;
  int pnbase;
  int my_port;
  opt.get("--portnumbase")->getInt(pnbase);
  opt.get("--hostname")->getString(hostname);
  opt.get("--ip-file-name")->getString(ipFileName);
  ez::OptionGroup* mp_opt = opt.get("--my-port");
  if (mp_opt->isSet)
    mp_opt->getInt(my_port);
  else
    my_port = Names::DEFAULT_PORT;

  Names playerNames;

  if (ipFileName.size() > 0) {
    if (my_port != Names::DEFAULT_PORT)
      throw runtime_error("cannot set port number when using IP file");
    playerNames.init(playerno, pnbase, ipFileName, nplayers);
  } else {
    Server::start_networking(playerNames, playerno, nplayers,
                             hostname, pnbase, my_port);
  }
  player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
}


void TreeInferenceServer::send_single_answer(Ciphertext &answer) {
  std::ostringstream stream;
  answer.save(stream);
  octetStream os(stream.str());
  player->send(os);

  
}


void TreeInferenceServer::run() {
  read_tree_structure_and_feature_info();
  recv_keys();
  timer.start(player->total_comm());

  vector<EncryptedSample> samples;
  cout << "receiving queries" << endl;
  for (int i = 0; i < test_sample_number; i++){
    EncryptedSample sample;
    recv_single_query(sample.features_vector);
    samples.push_back(sample);
  }
  cout << "processing queries" << endl;
  for (int i = 0; i < test_sample_number; i++){
    samples[i].label = process_single_query(samples[i].features_vector);
  }
  for (int i = 0; i < test_sample_number; i++){
    send_single_answer(samples[i].label);
  }

  timer.stop(player->total_comm());
  cout << "Server total time = " << timer.elapsed() << " seconds" << endl;
  cout << "Server data sent = " << timer.mb_sent() << " MB";
}


void TreeInferenceServer::recv_single_query(vector<vector<Ciphertext> > &features_vector) {
  for (int i = 0; i < feature_number; i++){
    if (max_values[i] < value_max_threhold) {
      vector<Ciphertext> onehot;
      for (int j = 0; j < max_values[i]; j++) {
        Ciphertext c;
        onehot.push_back(c);
      }
      features_vector.push_back(onehot);
    }else{
      int row = ceil(sqrt(max_values[i]));
      int col = ceil(max_values[i] / row);
      vector<Ciphertext> row_and_col_vector;
      for (int i = 0; i < row + col; i++){
        Ciphertext c;
        row_and_col_vector.push_back(c);
      }
      features_vector.push_back(row_and_col_vector);
    }
  }
  octetStream os;
  Timer recv_timer;
  recv_timer.start();
  player->receive(os);
  recv_timer.stop();
//  cout << "Recv time = " << recv_timer.elapsed() << " seconds" << endl;
  string s = os.str();
  std::istringstream stream(s);

  for (int i = 0; i < feature_number; i++){
    int size = features_vector[i].size();
    for (int j = 0; j < size; j++){
      features_vector[i][j].load(*context, stream);
    }
  }
}

Ciphertext TreeInferenceServer::process_single_query(vector<vector<Ciphertext> > &features_vector) {

  int tree_number = roots.size();

  PredictValue* root_values[tree_number];
  for (int i = 0; i < tree_number; i++){
    root_values[i] = roots[i]->predict(features_vector);

  }
  Ciphertext answer = root_values[0]->get_final_result();
  for (int i = 1; i < tree_number; i++){
    evaluator->add_inplace(answer, root_values[i]->get_final_result());
  }

  return answer;
}

void TreeInferenceServer::read_tree_structure_and_feature_info() {
  std::ifstream meta_file ("Player-Data/xgboost-meta");
  meta_file >> tree_number;
  meta_file >> tree_h;
  meta_file >> feature_number;
  meta_file >> test_sample_number;
  for (int i = 0; i < feature_number; i++){
    int max_value;
    meta_file >> max_value;
    max_values.push_back(max_value);
  }
  meta_file.close();
  std::ifstream node_file ("Player-Data/Input-P0-0");


  vector<Node*> node_of_last_layer;
  vector<Node*> node_of_current_layer;

  int max_node_number = 1 << tree_h;
  for (int j = 0; j < max_node_number; j++){
    node_of_last_layer.push_back(nullptr);
    node_of_current_layer.push_back(nullptr);
  }

  for (int k = 0; k < tree_number; k++){
      // first layer, root node
      int node_id, attr_id, threhold;
      node_file >> node_id;
      node_file >> attr_id;
      node_file >> threhold;
      Node* root = new Node(attr_id, threhold, false);
      roots.push_back(root);
      node_of_last_layer[0] = root;
      // internal layer
      for (int i = 1; i < tree_h; i++){
        int node_number = 1 << i;
        for (int j = 0; j < max_node_number; j++){
          node_of_current_layer[j] = nullptr;
        }
        for (int j = 0; j < node_number; j++){
          node_file >> node_id;
          node_file >> attr_id;
          node_file >> threhold;
          if (node_id < 0)
            continue;
          Node* node = new Node(attr_id, threhold, false);
          node_of_current_layer[node_id] = node;
        }
        int node_number_of_last_layer = 1 << (i - 1);
        for (int j = 0; j < node_number_of_last_layer; j++){
          if (node_of_last_layer[j] != nullptr) {
            node_of_last_layer[j]->left_child_node = node_of_current_layer[j];
            node_of_last_layer[j]->right_child_node = node_of_current_layer[j + node_number_of_last_layer];
          }
        }
        for (int j = 0; j < node_number; j++){
          node_of_last_layer[j] = node_of_current_layer[j];
        }
      }
      // leaf layer
      double label;
      int node_number = 1 << tree_h;
      for (int j = 0; j < max_node_number; j++){
        node_of_current_layer[j] = nullptr;
      }
      for (int j = 0; j < node_number; j++){
        node_file >> node_id;
        node_file >> label;
        if (node_id < 0)
            continue;
        int scale_label = round(label*scale_for_decimal_part);
        scale_label = (scale_label + plain_modulus) % plain_modulus;
        Node* node = new Node(attr_id, scale_label, true);
        node_of_current_layer[node_id] = node;
      }
      int node_number_of_last_layer = 1 << (tree_h - 1);
      for (int j = 0; j < node_number_of_last_layer; j++){
        if (node_of_last_layer[j] != nullptr) {
          node_of_last_layer[j]->left_child_node = node_of_current_layer[j];
          node_of_last_layer[j]->right_child_node = node_of_current_layer[j + node_number_of_last_layer];
        }
      }
  }
  for (int i = 0 ;i < tree_number; i++){
    merge_node(roots[i]);
  }
  node_file.close();

}

void TreeInferenceServer::recv_keys() {
  octetStream os;
  player->receive(os);
  string s = os.str();
  std::istringstream stream(s);
  public_key = new PublicKey();
  public_key->load(*context, stream);
  relin_keys = new RelinKeys();
  relin_keys->load(*context, stream);
  encryptor = new Encryptor(*context, *public_key);
//  // only for test
//  SecretKey secretKey;
//  secretKey.load(*context, stream);
//  decryptor_for_test = new Decryptor(*context, secretKey);
}

void TreeInferenceServer::merge_node(Node *node) {
  if (node->is_leaf)
    return;
  if (node->left_child_node != nullptr)
    merge_node(node->left_child_node);
  if (node->right_child_node != nullptr)
    merge_node(node->right_child_node);
  if (node->threhold < 0){
    delete node->left_child_node;
    node->left_child_node = node->right_child_node->left_child_node;
    Node* right_child = node->right_child_node;
    node->feature_id = right_child->feature_id;
    node->threhold = right_child->threhold;
    node->label = right_child->label;
    node->is_leaf = right_child->is_leaf;
    node->right_child_node = right_child->right_child_node;
    delete right_child;
  }
}