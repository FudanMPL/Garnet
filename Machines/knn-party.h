//
// Created by 周瑞生 on 2024/4/28.
//

#ifndef GARNET_KNNPARTY_H
#define GARNET_KNNPARTY_H

#include <iostream>
#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "../Tools/int.h"
#include "Tools/TimerWithComm.h"
#include "../Math/bigint.h"
#include "../Math/Z2k.h"
#include "../Machines/knn-party.hpp"

using namespace std;
typedef unsigned long size_t;
// template<size_t K>
// class additive
// {
//     Z2<K>share;
//     additive(Z2<K>&value)
//     additive operator+(const additive&other)
//     {
//         additive result(this->share+other.share);
//         return result;
//     }
//     bool operator<(const additive&other)
//     {

//     }
// };


class KNN_party
{
public:
    TimerWithComm timer;
    int playerno = 0;//player编号
    const int nplayers = 2;
    int num_features;// 特征数
    int num_train_data; // 样本数据总量
    RealTwoPartyPlayer* player; // 通信

    // const unsigned char* seed_constant='1';
    // PRNG seed();

    KNN_party(int playerNo):playerno(playerNo){};//构造函数
    void start_networking(ez::ezOptionParser& opt);//建立连接

    void send_single_query(  vector<Z2<64>> &query   );
    int recv_single_answer();




    vector<octetStream>data_send;
    vector<octetStream> data_receive;
    
    // vector< vector<  Z2<K>  > > triples( num_features, vector< Z2<K> >(3));
    void load_triples(string file_path);
   
   
    // void get_input_from(int palyerno,bool has_label,string file_path);//输入数据、、
    // vector<aby2> share_data_aby2();
    // vector<additive> compute_ESD(vector<aby2>&X,vector<aby2>&Y);
    void run();


};


#endif //GARNET_KNNPARTY_H