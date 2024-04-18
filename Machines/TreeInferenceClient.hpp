//
// Created by 林国鹏 on 2023/6/2.
//

#include "TreeInferenceClient.h"
#include "TreeInferenceServer.h"


TreeInferenceClient::TreeInferenceClient() {
  EncryptionParameters parms(scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  parms.set_plain_modulus(plain_modulus);
  context = new SEALContext(parms);
  keygen = new KeyGenerator(*context);
  secret_key = &keygen->secret_key();
  public_key = new PublicKey();
  keygen->create_public_key(*public_key);
  relin_keys = new RelinKeys();
  keygen->create_relin_keys(*relin_keys);
  encryptor = new Encryptor(*context, *public_key);
  evaluator = new Evaluator(*context);
  decryptor = new Decryptor(*context, *secret_key);
}

void TreeInferenceClient::start_networking(ez::ezOptionParser &opt) {
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


void TreeInferenceClient::run() {
  read_meta_and_sample();
  timer.start(player->total_comm());
  send_keys();
  for (int i = 0; i < test_sample_number; i++){
    vector<vector<Ciphertext> > query;
    generate_single_query(samples[i].features,  query);
    send_single_query(query);
  }
  double right = 0;
  for (int i = 0; i < test_sample_number; i++){
    int answer = recv_single_answer();
    cout << "recv answer = "  << answer << endl;
    if (answer > int(plain_modulus / 2))
      answer = answer - plain_modulus;
    int label = round(answer * 1.0 / scale_for_decimal_part);
    cout << "pred label =  "  << label << " real label = " <<  samples[i].label << endl;
    right += label == samples[i].label;
  }
  cout << "test accuracy: " << right / test_sample_number << endl;

  timer.stop(player->total_comm());
  cout << "Client total time = " << timer.elapsed() << " seconds" << endl;
  cout << "Client Data sent = " << timer.mb_sent() << " MB";

}

void TreeInferenceClient::read_meta_and_sample() {
  std::ifstream meta_file ("Player-Data/xgboost-meta");
  meta_file >> tree_number;
  meta_file >> tree_h;
  meta_file >> feature_number;
  meta_file >> test_sample_number;
  for (int i = 0; i < feature_number; i++){
    int temp;
    meta_file >> temp;
    max_values.push_back(temp);
  }
  meta_file.close();
  std::ifstream sample_file ("Player-Data/Input-P1-0");
  for (int i = 0; i < test_sample_number; i++){
    Sample sample;
    sample_file >> sample.label;
    samples.push_back(sample);
  }
  for (int i = 0; i < feature_number; i++){
    for (int j = 0; j < test_sample_number; j++){
      int feature;
      sample_file >> feature;
      samples[j].features.push_back(feature);
    }
  }
  sample_file.close();
//  features.push_back(10);
//  features.push_back(20);
//  features.push_back(30);
//  max_values.push_back(100);
//  max_values.push_back(100);
//  max_values.push_back(100);
}

//void TreeInferenceClient::generate_query(vector<int> &features, vector<int> &max_values, vector<vector<Ciphertext> >& query) {
//  int size = features.size();
//  const std::uint64_t temp_1 = 1;
//  const std::uint64_t temp_0 = 0;
//  Plaintext one(seal::util::uint_to_hex_string(&temp_1, std::size_t(1)));
//  Plaintext zero(seal::util::uint_to_hex_string(&temp_0, std::size_t(1)));
//  for (int i = 0; i < size; i++){
//    vector<Ciphertext> onehot;
//    for (int j = 0; j < max_values[i]; j++){
//      if (features[i] == j){
//        Ciphertext x_encrypted;
//        encryptor->encrypt(one, x_encrypted);
//        onehot.push_back(x_encrypted);
//      }else{
//        Ciphertext x_encrypted;
//        encryptor->encrypt(zero, x_encrypted);
//        onehot.push_back(x_encrypted);
//      }
//    }
//    query.push_back(onehot);
//  }
//}

void TreeInferenceClient::generate_single_query(vector<int> &features,  vector<vector<Ciphertext> >& query) {
  int size = features.size();
  const std::uint64_t temp_1 = 1;
  const std::uint64_t temp_0 = 0;
  Plaintext one(seal::util::uint_to_hex_string(&temp_1, std::size_t(1)));
  Plaintext zero(seal::util::uint_to_hex_string(&temp_0, std::size_t(1)));
  for (int i = 0; i < size; i++){

    if (max_values[i] < value_max_threhold){
      vector<Ciphertext> prefix_sum;
      for (int j = 0; j < max_values[i]; j++){
        if (features[i] <= j){
          Ciphertext x_encrypted;
          encryptor->encrypt(one, x_encrypted);
          prefix_sum.push_back(x_encrypted);
        }else{
          Ciphertext x_encrypted;
          encryptor->encrypt(zero, x_encrypted);
          prefix_sum.push_back(x_encrypted);
        }
      }
      query.push_back(prefix_sum);
    }
    else{
      vector<Ciphertext> row_and_col_vector;
      int row = ceil(sqrt(max_values[i]));
      int col = ceil(max_values[i] / row);
      int row_index = features[i] / col;
      int col_index = features[i] - (row_index * col);
//      cout << row << " " << col << endl;
//      cout << row_index << " " << col_index << endl;
      for (int j = 0; j < col; j++){
        if (j == col_index){
          Ciphertext x_encrypted;
          encryptor->encrypt(one, x_encrypted);
          row_and_col_vector.push_back(x_encrypted);
        } else{
          Ciphertext x_encrypted;
          encryptor->encrypt(zero, x_encrypted);
          row_and_col_vector.push_back(x_encrypted);
        }
      }
      for (int j = 0; j < row; j++){
        if (j == row_index){
          Ciphertext x_encrypted;
          encryptor->encrypt(one, x_encrypted);
          row_and_col_vector.push_back(x_encrypted);
        } else{
          Ciphertext x_encrypted;
          encryptor->encrypt(zero, x_encrypted);
          row_and_col_vector.push_back(x_encrypted);
        }
      }
      query.push_back(row_and_col_vector);
    }
  }
}

void TreeInferenceClient::send_single_query(vector<vector<Ciphertext> > &query) {
  std::ostringstream stream;

  int size = query.size();
  for (int i = 0; i < size; i++){
    int size2 = query[i].size();
    for (int j = 0; j < size2; j++){
      query[i][j].save(stream);
    }
  }
  octetStream os(stream.str());
  player->send(os);
}

void TreeInferenceClient::send_keys() {
  std::ostringstream stream;
  public_key->save(stream);
  relin_keys->save(stream);
//  secret_key->save(stream);
  octetStream os(stream.str());
  player->send(os);
}


int TreeInferenceClient::recv_single_answer() {
  octetStream os;
  player->receive(os);
  string s = os.str();
  std::stringstream received_ciphertext_stream(s);
  seal::Ciphertext received_ciphertext;
  received_ciphertext.load(*context, received_ciphertext_stream);
  int budget = decryptor->invariant_noise_budget(received_ciphertext);
  if (budget <= 0){
    cout << "  noise budget in the recv answer: " << budget << " bits, error may be wrong."
       << endl;
  }

  Plaintext answer;
  decryptor->decrypt(received_ciphertext, answer);

  return std::stoi(answer.to_string(), nullptr, 16);
}