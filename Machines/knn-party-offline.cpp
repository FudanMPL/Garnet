#include <iostream>
#include <random>
#include <chrono>
#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "Tools/PointerVector.h"
#include "../Tools/int.h"
#include "../Math/bigint.h"
#include "../Math/Z2k.hpp"
#include "Tools/TimerWithComm.h"
#include "Math/FixedVec.h"

using namespace std;
const int K=64;
void gen_fake_dcf(int beta, int n);
void generate_triples_save_file_optimized(); //dealer方生成所有aby2 share随机数，自定义的三元组数据，并存入到对应文件中。属于set-up阶段，运行一次，后续就不用再运行了。

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int m_playerno = 0;//player编号
int num_features;// 特征数
int num_train_data; // 训练集数据总量
int num_test_data; // 测试集数据总量
int num_label; // 训练集中label数量
string dataset_name="chronic";
// string dataset_name="mnist";//数据集名称，自动用于后续的文件名生成

void read_meta_data()
{
    string file_meta_file="./Player-Data/Knn-Data/"+dataset_name+"-data/Knn-meta";
    if(fileExists(file_meta_file))
    {
        std::ifstream meta_file(file_meta_file);
        meta_file >> num_features;// 特征数
        meta_file >> num_train_data;//训练集数据大小
        meta_file >> num_test_data;//测试集数据大小
        meta_file >> num_label; //label 个数
        // m_label_list.resize(num_label);
        // for(int i=0;i<num_label;i++)meta_file >>m_label_list[i];
        meta_file.close();
    }
    else{
        cerr<<"Knn-meta file Not Exist!"<<endl;
    }
        
}





int main()
{
    // if(fileExists("Player-Data/k0"))
    // {
    //     cout<<"DCF_FILE_ALREADY_EXIST"<<endl;
    // }
    // else{
        cout<<"----GEN_FAKE_DCF_KEY BEGINNING------"<<endl;
        gen_fake_dcf(1,K);
        cout<<"----GEN_FAKE_DCF_KEY ENDDING------"<<endl;
    // }

    cout<<"----GEN_OPTIMIZED_TRIPLE_DATA BEGINNING------"<<endl;
    read_meta_data();
    generate_triples_save_file_optimized();
    cout<<"----GEN_OPTIMIZED_TRIPLE_DATA ENDDING------"<<endl;

    return 0;
}



void generate_triples_save_file_optimized()
{
    PRNG seed;
    seed.ReSeed();
    vector<vector<Z2<K>>>Train_Triples_0(num_train_data,vector<Z2<K>>(num_features,Z2<K>(0)) );
    vector<vector<Z2<K>>>Test_Triples_0(num_train_data+1,vector<Z2<K>>(num_features,Z2<K>(0)) );
    vector<vector<Z2<K>>>Train_Triples_1(num_train_data,vector<Z2<K>>(num_features,Z2<K>(0)) );
    vector<vector<Z2<K>>>Test_Triples_1(num_train_data+1,vector<Z2<K>>(num_features,Z2<K>(0)) );
    ofstream file_Train_Triples_0("./Player-Data/Knn-Data/"+dataset_name+"-data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/Knn-Data/"+dataset_name+"-data/P0-Test-Triples", std::ios::binary),
        file_Train_Triples_1("./Player-Data/Knn-Data/"+dataset_name+"-data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/Knn-Data/"+dataset_name+"-data/P1-Test-Triples", std::ios::binary);
    
    for(int i=0;i<num_features;i++)
    {
        Test_Triples_0[0][i].randomize(seed);
        Test_Triples_1[0][i].randomize(seed);
        file_Test_Triples_0.write(reinterpret_cast<char*>(&Test_Triples_0[0][i]), sizeof(Z2<K>));
        file_Test_Triples_1.write(reinterpret_cast<char*>(&Test_Triples_0[0][i]), sizeof(Z2<K>));
    }
    for(int i=0;i<num_features;i++)
        file_Test_Triples_1.write(reinterpret_cast<char*>(&Test_Triples_1[0][i]), sizeof(Z2<K>));


    for(int i=0;i<num_train_data;i++)
    {
        for(int j=0;j<num_features;j++)
        {
            Train_Triples_0[i][j].randomize(seed);
            Train_Triples_1[i][j].randomize(seed);
            Z2<K>tmp;
            tmp.randomize(seed);
            Test_Triples_0[i+1][j]=(Test_Triples_0[0][j]+Test_Triples_1[0][j]-Train_Triples_0[i][j]-Train_Triples_1[i][j])*
                                    (Test_Triples_0[0][j]+Test_Triples_1[0][j]-Train_Triples_0[i][j]-Train_Triples_1[i][j]) -tmp;
            Test_Triples_1[i+1][j]=tmp;
            file_Train_Triples_0.write(reinterpret_cast<char*>(&Train_Triples_0[i][j]), sizeof(Z2<K>));
            file_Train_Triples_1.write(reinterpret_cast<char*>(&Train_Triples_1[i][j]), sizeof(Z2<K>));
            file_Test_Triples_0.write(reinterpret_cast<char*>(&Test_Triples_0[i+1][j]), sizeof(Z2<K>));
            file_Test_Triples_1.write(reinterpret_cast<char*>(&Test_Triples_1[i+1][j]), sizeof(Z2<K>));
        }
    }
    file_Train_Triples_0.close();
    file_Train_Triples_1.close();
    file_Test_Triples_0.close();
    file_Test_Triples_1.close();
}   


void gen_fake_dcf(int beta, int n)
{
   // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = 16;
    PRNG prng;
    prng.InitSeed();
    fstream k0, k1, r0, r1, r2;
    k0.open("Player-Data/k0", ios::out);
    k1.open("Player-Data/k1", ios::out);
    r0.open("Player-Data/r0", ios::out);
    r1.open("Player-Data/r1", ios::out);
    r2.open("Player-Data/r2", ios::out);
    octet seed[2][lambda_bytes];    
    bigint s[2][2], v[2][2],  t[2][2], tmp_t[2], convert[2], tcw[2], a, scw, vcw, va, tmp, tmp1, tmp_out;
    prng.InitSeed();
    prng.get(tmp, n);
    bytesFromBigint(&seed[0][0], tmp, lambda_bytes);
    k0 << tmp << " ";
    prng.get(tmp1, n);
    bytesFromBigint(&seed[1][0], tmp1, lambda_bytes);
    k1 << tmp1 << " ";
    prng.get(a, n);
    prng.get(tmp, n);
    r1 << a - tmp << " ";
    r0 << tmp << " ";
    r2 << tmp << " ";
    r0.close();
    r1.close();
    r2.close();
    tmp_t[0] = 0;
    tmp_t[1] = 1;
    int keep, lose;
    va = 0;
    //We can optimize keep into one bit here
    // generate the correlated word!
    for(int i = 0; i < n - 1; i++){
        keep = bigint(a >>( n - i - 1)).get_ui() & 1;
        lose = 1^keep;
        for(int j = 0; j < 2; j++){     
            prng.SetSeed(seed[j]);
            // k is used for left and right
            for(int k = 0; k < 2; k++){
                prng.get(t[k][j], 1);
                prng.get(v[k][j], n);
                prng.get(s[k][j] ,n);
            }
        }
        scw = s[lose][0] ^ s[lose][1]; 
        // save convert(v0_lose) into convert[0]
        bytesFromBigint(&seed[0][0], v[lose][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], n);     
        // save convert(v1_lose) into convert[1]
        bytesFromBigint(&seed[0][0], v[lose][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], n);
        if(tmp_t[1])
            vcw = convert[0] + va - convert[1];
        else
            vcw = convert[1] - convert[0] - va;
        //keep == 1, lose = 0，so lose = LEFT
        if(keep)
            vcw = vcw + tmp_t[1]*(-beta) + (1-tmp_t[1]) * beta;
        // save convert(v0_keep) into convert[0]
        bytesFromBigint(&seed[0][0], v[keep][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], n);
        // save convert(v1_keep) into convert[1]
        bytesFromBigint(&seed[0][0], v[keep][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], n);
        va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1-tmp_t[1]) * vcw;
        tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
        tcw[1] = t[1][0] ^ t[1][1] ^ keep;
        k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        bytesFromBigint(&seed[0][0],  s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
        bytesFromBigint(&seed[1][0],  s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
        bigintFromBytes(tmp_out, &seed[0][0], lambda_bytes);
        bigintFromBytes(tmp_out, &seed[1][0], lambda_bytes);
        tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
        tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
    }
    
    prng.SetSeed(seed[0]);
    prng.get(convert[0], n);
    prng.SetSeed(seed[1]);
    prng.get(convert[1], n);
    k0 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k1 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
    k0.close();
    k1.close();

    return;
}
