//
// Created by 周瑞生 on 2024/4/28.
//

#ifndef GARNET_KNNPARTY_H
#define GARNET_KNNPARTY_H

#include <iostream>

using namespace std;

#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "../Tools/int.h"
#include "Tools/TimerWithComm.h"



class KNN_party
{
public:
    TimerWithComm timer;
    const int playerno = 0;
    const int nplayers = 2;
    int num_features;
    int num_train_data;
    RealTwoPartyPlayer* player;

    // vector< vector<  Z2<K>  > > triples( num_features, vector< Z2<K> >(3));

    void load_triples(string file_path);

    void start_networking(ez::ezOptionParser& opt);//建立连接
    // void get_input_from(int palyerno,bool has_label,string file_path);//输入数据、、
    // vector<aby2> share_data_aby2();
    // vector<additive> compute_ESD(vector<aby2>&X,vector<aby2>&Y);
    
    
};
#endif //GARNET_KNNPARTY_H