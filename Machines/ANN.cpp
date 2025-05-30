#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <numeric>
#include <omp.h>

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
void test_Z2();
const int K=64;//环大小
const int k_const=5;//knn里面的k值 

int k_left=5,k_right=30;
string dataset_name;
string dir;
typedef vector<int>Vec;
typedef  long long ll;
// string dataset_name="chronic";//数据集名称，自动用于后续的文件名生成
// string dataset_name="mnist";//数据集名称，自动用于后续的文件名生成

int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;
void parse_argv(int argc, const char** argv);
void gen_fake_dcf(int beta, int n);
bigint evaluate(Z2<K> x, int n,int playerID);
long long call_evaluate_time=0;


// 全局变量用于累计总运行时间
std::chrono::duration<double> total_duration(0);
class Sample {
public:
    Vec features;
    int label;
    Sample() : label(0) {}
    Sample(int n) : features(n),label(0) {};
    Sample(Vec features_,int label_):features(features_),label(label_){};
};

struct Kmeans_Result {
    int num_features, num_train_data, num_test_data;
    vector<Sample> m_sample; // 训练集
    vector<Sample> m_test;   // 测试集

    Kmeans_Result(int features, int train_num, int test_num,
                  const vector<Sample>& samples,
                  const vector<Sample>& tests)
        : num_features(features),
          num_train_data(train_num),
          num_test_data(test_num),
          m_sample(samples),
          m_test(tests) {}

    Kmeans_Result() : num_features(0), num_train_data(0), num_test_data(0) {}

  
};


class KNN_party_base
{
public:
    Kmeans_Result result;               // ✅ 保存返回的结构体
    typedef Z2<K> additive_share;
    TimerWithComm timer;
    const int nplayers=2;
    int m_playerno = 0;//player编号
    int num_features;// 特征数
    int num_train_data; // 训练集数据总量
    int num_test_data; // 测试集数据总量
    int num_label; // 训练集中label数量

    RealTwoPartyPlayer* m_player; // 通信模块
    vector<Sample*>m_sample; //训练集
    vector<Sample*>m_test; //测试集
    vector<int>m_label_list; //训练集中label的值列表

    vector< array<additive_share,2> >m_ESD_vec;
    vector<array<additive_share,2>>m_shared_label_list_count_array;
    virtual void run()=0;


    KNN_party_base(int playerNo):m_playerno(playerNo){};//构造函数

    void start_networking(ez::ezOptionParser& opt);//建立连接
    void read_meta_and_P0_sample_P1_query();
    virtual void compute_ESD_for_one_query(int idx_of_test)=0;
    virtual void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last)=0;  // top-1算法

    Z2<K> secure_compare(Z2<K>x1,Z2<K>x2,bool greater_than=true);//默认为x1>x2-->1 x1>x2-->0  x1=x2-->0 ******
    /*
        shares_selected_k为k个选择出来的share态的neighbors，label_list_count_array为二维数组，k_const*2,每一行第一个元素为出现次数，第二个元素为shares_selected_k[行表]
    */
    void label_compute(vector<Z2<K>>&shares_selected_k, vector<array<Z2<K>,2>>&label_list_count_array); 

    void compare_in_vec(vector<Z2<K>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than); //
    void compare_in_vec(vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than);

    Z2<K> reveal_one_num_to(Z2<K> x,int playerID);
    SignedZ2<K> reveal_one_num_to(SignedZ2<K> x,int playerID);

    void additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec={});
    void additive_share_data_vec(vector<Z2<K>>&shares);

    /* 
    加法秘密共享的数据向量乘：
    double_res为false: v1 * v2 --> res
    double_res为true: v1.size()和res.size()一致，等于2* v2.size()    v1[:half]*v2 || v1[half:]*v2 --> res 
     */
    void mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res , bool double_res);

    /*
    加法秘密共享的数据标量乘法：
    res=x1*x2
    */
    void mul_additive(Z2<K>x1,Z2<K>x2,Z2<K>&res);

    /*
    secure sort in vector:
    size: compare_idx_vec一定是2的倍数，每两个为一组，表示当前需要比较的元素的索引 ，compare_res：表示比较的值，为了对齐，方便乘法运算，也为2的倍数，重复一遍。
    shares: data to be sorted，如果是二维的array也是按照第一个维度数据来进行secure sort
    compare_idx_vec: 
    compare_res：
    */
    // void SS_vec( vector<Z2<K>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res);
    void SS_vec( vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res);

    /*
        secure sort in scalar
    */
    void SS_scalar(vector<Z2<K>>&shares,int first_idx,int second_idx,bool min_then_max=true);
    void SS_scalar(vector<array<Z2<K>,2>>&shares,int first_idx,int second_idx,bool min_then_max=true);

    
    void LabelCompute_test();
    void test_cmp();

};


class KNN_party_optimized:public KNN_party_base
{
public:
    
    typedef FixedVec<Z2<K>,2> aby2_share;
    vector<vector<aby2_share>>m_train_aby2_share_vec;
    vector<vector<aby2_share>>m_test_aby2_share_vec;
    vector<vector< Z2<K> > >m_Train_Triples_0;  //P0 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> > >m_Train_Triples_1;  //P1 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> >>m_Test_Triples; // num_train_data * num_features  个三元组的第三个值：[(\delta_x - \delta_y)*(\delta_x - \delta_y)]
    vector< Z2<K> > m_Test_Triples_0;   // num_features 个随机数，P0用于aby2 share
    vector< Z2<K> > m_Test_Triples_1;  // num_features 个随机数，P1用于aby2 share

    KNN_party_optimized(int playerNo):KNN_party_base(playerNo){
        std::cout<<"Entering the KNN_party_optimized class:"<<std::endl;
    }

    void generate_triples_save_file(); //dealer方生成所有aby2 share随机数，自定义的三元组数据，并存入到对应文件中。属于set-up阶段，运行一次，后续就不用再运行了。
    
    void load_triples(); //读入三元组数据 分别： P0:m_Train_Triples_0 m_Train_Triples_1(aby2share的share协议) m_Test_Triples_0   P1：m_Train_Triples_1, m_Test_Triples_0, m_Test_Triples_1
    void fake_load_triples();//fake形式读入三元组数据 分别： P0:m_Train_Triples_0 m_Train_Triples_1(aby2share的share协议) m_Test_Triples_0   P1：m_Train_Triples_1, m_Test_Triples_0, m_Test_Triples_1

    void aby2_share_data_and_additive_share_label_list(); //把训练和测试数据分别在P0,P1使用aby2 share协议进行share,并且将label数据转换成加法秘密共享状态
    void aby2_share_reveal(int idx,bool is_sample_data); //测试使用，idx为reveal的样本的索引

    void additive_share_label_data();//

    Z2<K> compute_ESD_two_sample(int idx_of_sample,int idx_of_test);

    void compute_ESD_for_one_query(int idx_of_test);

    

    void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last=true);

    void run();

};



void save_result_to_file(const Kmeans_Result& result, const string& filename) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    out << result.num_features << " " << result.num_train_data << " " << result.num_test_data << endl;

    // 保存训练集
    for (const auto& sample : result.m_sample) {
        for (int val : sample.features)
            out << val << " ";
        out << sample.label << endl;
    }

    // 保存测试集
    for (const auto& sample : result.m_test) {
        for (int val : sample.features)
            out << val << " ";
        out << sample.label << endl;
    }

    out.close();
    cout << "Saved Kmeans_Result to " << filename << endl;
}

Kmeans_Result load_result_from_file(const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Failed to open file for reading: " << filename << endl;
        exit(1);
    }

    Kmeans_Result result;
    in >> result.num_features >> result.num_train_data >> result.num_test_data;

    // 读取训练集
    for (int i = 0; i < result.num_train_data; ++i) {
        Vec features(result.num_features);
        for (int j = 0; j < result.num_features; ++j) {
            in >> features[j];
        }
        int label;
        in >> label;
        result.m_sample.emplace_back(features, label);
    }

    // 读取测试集
    for (int i = 0; i < result.num_test_data; ++i) {
        Vec features(result.num_features);
        for (int j = 0; j < result.num_features; ++j) {
            in >> features[j];
        }
        int label;
        in >> label;
        result.m_test.emplace_back(features, label);
    }

    in.close();
    return result;
}



int main(int argc, const char** argv)
{
    parse_argv(argc, argv);
    dir="knn-1/";
    // vector<string>dataset_name_list={"Iris","Wine","Cancer","Spambase","Adult","Mnist","Dota2Games", "Toxicity", "arcene", "RNA-seq", "PEMS-SF"};
    vector<string>dataset_name_list={"Mnist"};
    for(int i=0;i<dataset_name_list.size();i++){
        dataset_name=dataset_name_list[i];
        cout<<"--------DataSet:"<<dataset_name<<"--------------"<<endl;
        KNN_party_optimized party(playerno);
        party.start_networking(opt);
        std::cout<<"Network Set Up Successful ! "<<std::endl;
        party.run();
    }
    return 0;
}

void KNN_party_optimized::aby2_share_data_and_additive_share_label_list()
{
    
    //aby2_share_data
    cout<<"num_train_data:"<<num_train_data<<endl;
    m_train_aby2_share_vec.resize(num_train_data);
    for(int i=0;i<num_train_data;i++)
        m_train_aby2_share_vec[i].resize(num_features);
    
    m_test_aby2_share_vec.resize(num_test_data);
    for(int i=0;i<num_test_data;i++)
        m_test_aby2_share_vec[i].resize(num_features);
    if(m_playerno==0)
    {
        octetStream os;
        for(int i=0;i<num_train_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {   
                m_train_aby2_share_vec[i][j][1]=m_Train_Triples_0[i][j];
                m_train_aby2_share_vec[i][j][0]=Z2<K>(m_sample[i]->features[j])+m_Train_Triples_0[i][j]+m_Train_Triples_1[i][j];
                m_train_aby2_share_vec[i][j][0].pack(os);
            }
        }
        m_player->send(os);
        // cout<<"Train data aby2_share sending ended!"<<endl;

        os.clear();
        m_player->receive(os);
        for(int i=0;i<num_test_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {
                m_test_aby2_share_vec[i][j][1] = m_Test_Triples_0[j];
                m_test_aby2_share_vec[i][j][0].unpack(os);
            }
        }
        // cout<<"Test data aby2_share receiving ended!"<<endl;

    }
    else
    {
        octetStream os;
        m_player->receive(os);
        for(int i=0;i<num_train_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {
                m_train_aby2_share_vec[i][j][1]=m_Train_Triples_1[i][j];
                m_train_aby2_share_vec[i][j][0].unpack(os);
            }
        }
        // cout<<"Train data aby2_share receiving ended!"<<endl;

        os.clear();
        for(int i=0;i<num_test_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {   
                m_test_aby2_share_vec[i][j][1]=m_Test_Triples_1[j];
                m_test_aby2_share_vec[i][j][0]=Z2<K>(m_test[i]->features[j])+m_Test_Triples_0[j]+m_Test_Triples_1[j];
                m_test_aby2_share_vec[i][j][0].pack(os);
            }
        }
        m_player->send(os);
        // cout<<"Test data aby2_share  sending ended!"<<endl;

    } 
}

void KNN_party_optimized::fake_load_triples()
{
    if(playerno==0)
    {
        m_Train_Triples_0.resize(num_train_data);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        m_Test_Triples_0.resize(num_features);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_0[i].resize(num_features);
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        // ifstream file_Train_Triples_0("./Player-Data/Knn-Data/"+dataset_name+"-data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/Knn-Data/"+dataset_name+"-data/P0-Test-Triples", std::ios::binary),
        //     file_Train_Triples_1("./Player-Data/Knn-Data/"+dataset_name+"-data/P1-Train-Triples", std::ios::binary);
        // for(int i=0;i<num_features;i++)
        // {
        //      file_Test_Triples_0.read(reinterpret_cast<char*>(&m_Test_Triples_0[i]), sizeof(Z2<K>));
        // }           
        // for(int i=0;i<num_train_data;i++)
        // {
        //     for(int j=0;j<num_features;j++)
        //     {
        //         file_Train_Triples_0.read(reinterpret_cast<char*>(&m_Train_Triples_0[i][j]), sizeof(Z2<K>));
        //         file_Train_Triples_1.read(reinterpret_cast<char*>(&m_Train_Triples_1[i][j]), sizeof(Z2<K>));
        //         file_Test_Triples_0.read(reinterpret_cast<char*>(&m_Test_Triples[i][j]), sizeof(Z2<K>));
        //     }
        // }
        // file_Train_Triples_0.close();
        // file_Train_Triples_1.close();
        // file_Test_Triples_0.close();
        // cout<<"P0 loading fake triple ended!"<<endl;
    }
    else
    {
        m_Test_Triples_0.resize(num_features);
        m_Test_Triples_1.resize(num_features);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        // ifstream file_Train_Triples_1("./Player-Data/Knn-Data/"+dataset_name+"-data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/Knn-Data/"+dataset_name+"-data/P1-Test-Triples", std::ios::binary);
        // for(int i=0;i<num_features;i++)
        // {
        //     file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_0[i]), sizeof(Z2<K>));
        //     // cout<<m_Test_Triples_0[i]<<" ";
        // }
        // // cout<<endl;
            
        // for(int i=0;i<num_features;i++)
        // {
        //     file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_1[i]), sizeof(Z2<K>));
        //     // cout<<m_Test_Triples_1[i]<<" ";
        // }
        // // cout<<endl;
            
        // for(int i=0;i<num_train_data;i++)
        // {
        //     for(int j=0;j<num_features;j++)
        //     {
        //         file_Train_Triples_1.read(reinterpret_cast<char*>(&m_Train_Triples_1[i][j]), sizeof(Z2<K>));
        //         file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples[i][j]), sizeof(Z2<K>));
        //     }
        // }
        // file_Train_Triples_1.close();
        // file_Test_Triples_1.close();

        // cout<<"P1 loading fake triple ended!"<<endl;
    }
}


void KNN_party_optimized::load_triples()
{
    if(playerno==0)
    {
        m_Train_Triples_0.resize(num_train_data);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        m_Test_Triples_0.resize(num_features);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_0[i].resize(num_features);
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        ifstream file_Train_Triples_0("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P0-Test-Triples", std::ios::binary),
            file_Train_Triples_1("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P1-Train-Triples", std::ios::binary);
        for(int i=0;i<num_features;i++)
        {
             file_Test_Triples_0.read(reinterpret_cast<char*>(&m_Test_Triples_0[i]), sizeof(Z2<K>));
        }           
        for(int i=0;i<num_train_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {
                file_Train_Triples_0.read(reinterpret_cast<char*>(&m_Train_Triples_0[i][j]), sizeof(Z2<K>));
                file_Train_Triples_1.read(reinterpret_cast<char*>(&m_Train_Triples_1[i][j]), sizeof(Z2<K>));
                file_Test_Triples_0.read(reinterpret_cast<char*>(&m_Test_Triples[i][j]), sizeof(Z2<K>));
            }
        }
        file_Train_Triples_0.close();
        file_Train_Triples_1.close();
        file_Test_Triples_0.close();
        // cout<<"P0 loading triple ended!"<<endl;
    }
    else
    {
        m_Test_Triples_0.resize(num_features);
        m_Test_Triples_1.resize(num_features);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        ifstream file_Train_Triples_1("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P1-Test-Triples", std::ios::binary);
        for(int i=0;i<num_features;i++)
        {
            file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_0[i]), sizeof(Z2<K>));
            // cout<<m_Test_Triples_0[i]<<" ";
        }
        // cout<<endl;
            
        for(int i=0;i<num_features;i++)
        {
            file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_1[i]), sizeof(Z2<K>));
            // cout<<m_Test_Triples_1[i]<<" ";
        }
        // cout<<endl;
            
        for(int i=0;i<num_train_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {
                file_Train_Triples_1.read(reinterpret_cast<char*>(&m_Train_Triples_1[i][j]), sizeof(Z2<K>));
                file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples[i][j]), sizeof(Z2<K>));
            }
        }
        file_Train_Triples_1.close();
        file_Test_Triples_1.close();

        // cout<<"P1 loading triple ended!"<<endl;
    }
}


void KNN_party_optimized::generate_triples_save_file()
{
    if(m_playerno==0)
    {
        PRNG seed;
        seed.ReSeed();
        vector<vector<Z2<K>>>Train_Triples_0(num_train_data,vector<Z2<K>>(num_features,Z2<K>(0)) );
        vector<vector<Z2<K>>>Test_Triples_0(num_train_data+1,vector<Z2<K>>(num_features,Z2<K>(0)) );
        vector<vector<Z2<K>>>Train_Triples_1(num_train_data,vector<Z2<K>>(num_features,Z2<K>(0)) );
        vector<vector<Z2<K>>>Test_Triples_1(num_train_data+1,vector<Z2<K>>(num_features,Z2<K>(0)) );
        ofstream file_Train_Triples_0("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P0-Test-Triples", std::ios::binary),
            file_Train_Triples_1("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/Knn-Data/"+dir+dataset_name+"-data/P1-Test-Triples", std::ios::binary);
        
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
    else
        return;
}   

void KNN_party_base::start_networking(ez::ezOptionParser& opt) 
{
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
    this->m_player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
    player = this->m_player;
  }


Z2<K> KNN_party_optimized::compute_ESD_two_sample(int train_idx,int query_idx)
{
    Z2<K>res(0);
    Z2<K>tmp_1(0);
    for(int j=0;j<num_features;j++)
    {
        aby2_share x_min_y_aby2_share=m_train_aby2_share_vec[train_idx][j]-m_test_aby2_share_vec[query_idx][j];
        tmp_1=tmp_1+x_min_y_aby2_share[0]*x_min_y_aby2_share[0];
        res=res-Z2<K>(2)*x_min_y_aby2_share[0]*x_min_y_aby2_share[1]+m_Test_Triples[train_idx][j];
    }
    if(playerno==1)
    {
        res=res+tmp_1;
        // octetStream os;
        // res.pack(os);
        // m_player->send(os);
    }
    else
    {
        // octetStream os;
        // Z2<K>res_recv;
        // m_player->receive(os);
        // res_recv.unpack(os);
        // cout<<"Revealed ESD="<<res_recv+res<<endl;
    }
    return res;

}

Z2<K> KNN_party_base::reveal_one_num_to(Z2<K> x,int playerID)
{
    octetStream os;
    if(playerno==playerID)
    {
        m_player->receive(os);
        Z2<K>tmp;
        tmp.unpack(os);
        return tmp+x;
    }
    else
    {
        x.pack(os);
        m_player->send(os);
        return x;
    }
}

SignedZ2<K> KNN_party_base::reveal_one_num_to(SignedZ2<K> x,int playerID)
{
    octetStream os;
    if(playerno==playerID)
    {
        m_player->receive(os);
        SignedZ2<K>tmp;
        tmp.unpack(os);
        return tmp+x;
    }
    else
    {
        x.pack(os);
        m_player->send(os);
        return x;
    }
}



void KNN_party_optimized::run()
{
    read_meta_and_P0_sample_P1_query();

    std::cout<<"sample size:"<<num_train_data<<std::endl;
    std::cout<<"test size:"<<num_test_data<<std::endl;
    std::cout<<"Feature size:"<<num_features<<std::endl;

    // generate_triples_save_file();//这个函数必须独立运行，不能和后续load_triple一起使用。
    // cout<<"\n generate_triples_save_file success!"<<endl;

    // generate_triples_save_file();

    // load_triples();
    fake_load_triples();
    aby2_share_data_and_additive_share_label_list();
    cout<<std::flush;
    timer.start(m_player->total_comm());

    player->VirtualTwoPartyPlayer_Round=0;


    int right_prediction_cnt=0;
    for(int idx=0 ; idx<num_test_data ; idx++)
    {
        // cout<<"==========================="<<endl;
        Z2<K>tmp0(0);

        compute_ESD_for_one_query(idx);
        // cout<<"Compute ESD' Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
        // for(int i=0;i<num_train_data;i++)
        // {
        //     tmp0=reveal_one_num_to(m_ESD_vec[i][0],0);
        //     if(m_playerno==0)std::cout<<tmp0<<" ";
        // }
        // std::cout<<std::endl;
        // std::cout<<std::endl;

        // 选择top-k 最小的k个值放到最后面的k个位置
        for(int i=0;i<k_const;i++){
            top_1(m_ESD_vec,num_train_data-i,true); 
            // cout<<"2  Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
        }

        // for(int i=0;i<num_train_data;i++)
        // {
        //     tmp0=reveal_one_num_to(m_ESD_vec[i][0],0);
        //     if(m_playerno==0)std::cout<<tmp0<<" ";
        // }
        // std::cout<<std::endl;
        // std::cout<<std::endl;
            
        // cout<<"top_1 Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
        vector<Z2<K>>shares_selected_k;
        for(int i=0;i<k_const;i++)shares_selected_k.push_back(m_ESD_vec[m_ESD_vec.size()-1-i][1]);
        
        this->label_compute(shares_selected_k,m_shared_label_list_count_array);
        // cout<<"label_compute  Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
        top_1(m_shared_label_list_count_array,k_const,false);
        // cout<<"top_1 of K neighbors  Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
        
        Z2<K>predicted_label=reveal_one_num_to(m_shared_label_list_count_array[k_const-1][1],1); 
        if(m_playerno)
        {
            if(Z2<K>(m_test[idx]->label)==predicted_label)
                right_prediction_cnt++;
        }

    }


    if(m_playerno)
        std::cout<<"\n预测准确率:"<<double(right_prediction_cnt)/(double)num_test_data<<endl;

    timer.stop(m_player->total_comm());
    cout<<"Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
    std::cout << "Party total time = " << timer.elapsed() << " seconds" << std::endl;
    std::cout << "Party Data sent = " << timer.mb_sent() << " MB"<<std::endl;

    std::cout<<"call_evaluate_nums : "<<call_evaluate_time<<std::endl;

    std::cout << "在Evaluation函数中 Total elapsed time: " << total_duration.count() << " seconds" << std::endl;
    call_evaluate_time=0;
    total_duration = std::chrono::duration<double>(0);

    std::cout<<"-----------------------------------------------------"<<std::endl;


}


void KNN_party_base::label_compute(vector<Z2<K>>&share_k_neighbors, vector<array<Z2<K>,2>>&label_list_count_array)
{
    label_list_count_array.resize(k_const);
    vector<vector<Z2<K>>> cmp_2d_vec_res(k_const,vector<Z2<K>>(k_const));
    vector<int>cmp_vec_idx;
    for(int i=0;i<k_const;i++){
        for(int j=0;j<k_const;j++){
            cmp_vec_idx.push_back(i);
            cmp_vec_idx.push_back(j);
        }
    }
    vector<Z2<K>>cmp_res_vec(cmp_vec_idx.size());
    compare_in_vec(share_k_neighbors,cmp_vec_idx,cmp_res_vec,true);
    for(int i=0;i<cmp_res_vec.size()/2;i++){
        cmp_2d_vec_res[cmp_vec_idx[2*i]][cmp_vec_idx[2*i+1]]=Z2<K>(m_playerno)-cmp_res_vec[2*i];
    }

    vector<Z2<K>>v1(k_const*k_const),v2(k_const*k_const),res(k_const*k_const);
    for(int i=0;i<k_const;i++){
        for(int j=0;j<k_const;j++){
            v1[i*k_const+j]=cmp_2d_vec_res[i][j];
            v2[i*k_const+j]=cmp_2d_vec_res[j][i];
        }
    }
    mul_vector_additive(v1,v2,res,false);

    // vector<array<Z2<K>,2>>label_list_count_array(k_const);
    for(int i=0;i<k_const;i++){
        Z2<K>tmp(0);
        for(int j=0;j<k_const;j++){
            tmp+=res[i*k_const+j];
        }
        label_list_count_array[i]={tmp,share_k_neighbors[i]};
    }
    // cout<<"label_list_count_array的测试结果(频率  label):"<<endl;
    // for(int i=0;i<k_const;i++){
    //     Z2<K>tmp_0=reveal_one_num_to(label_list_count_array[i][0],1);//出现频率
    //     Z2<K>tmp_1=reveal_one_num_to(label_list_count_array[i][1],1);//对应label
    //     if(playerno==1){
    //         std::cout<<tmp_0<<"  "<<tmp_1<<endl;
    //     }
    // }



    // for(int i=0;i<num_label;i++)
    // {
    //     Z2<K>tmp(0);
    //     for(int j=0;j<k_const;j++)
    //     {
    //         // Z2<K>u1=Z2<K>(m_playerno)-this->secure_compare(label_list_count_array[i][1],shares_selected_k[j]);
    //         // Z2<K>u2=Z2<K>(m_playerno)-this->secure_compare(shares_selected_k[j],label_list_count_array[i][1]);
    //         vector<Z2<K>>U(4),value_tmp={label_list_count_array[i][1],shares_selected_k[j],shares_selected_k[j],label_list_count_array[i][1]};
    //         compare_in_vec(value_tmp,{0,1,2,3},U,true);
    //         Z2<K>tmp_res;
    //         mul_additive(Z2<K>(m_playerno)-U[0],Z2<K>(m_playerno)-U[2],tmp_res);//这一步Z2<K>(m_playerno)-很重要,不然都是错误的
    //         tmp+=tmp_res;
    //     }
    //     label_list_count_array[i][0]=tmp;
    // }
    
}

void KNN_party_base::compare_in_vec(vector<Z2<K>>&shares,const vector<int>compare_idx_vec,vector<Z2<K>>&compare_res,bool greater_than)
{
    assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
    bigint r_tmp;
    fstream r;
    r.open("Player-Data/2-fss/r" + to_string( m_playerno), ios::in);
    r >> r_tmp;
    r.close();
    SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
    // cout<<"\n bigint:"<<r_tmp<<"  Z2<K>: " << alpha_share<<endl;
    int size_res=compare_idx_vec.size()/2;

    vector<SignedZ2<K>>compare_res_t(compare_res.size());
    if(greater_than)
    {
        for(int i=0;i<size_res;i++)
        {
            compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[2*i+1]])-SignedZ2<K>(shares[compare_idx_vec[2*i]])+alpha_share;
            //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
        }
    }
    else{
        for(int i=0;i<size_res;i++)
        {
            compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[2*i]])-SignedZ2<K>(shares[compare_idx_vec[2*i+1]])+alpha_share;
            //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
        }
    }

       
    vector<SignedZ2<K>>tmp_res(size_res);

    octetStream send_os,receive_os;
    for(int i=0;i<size_res;i++)compare_res_t[i].pack(send_os);
    player->send(send_os);
    player->receive(receive_os);
    for(int i=0;i<size_res;i++)
    {
        SignedZ2<K>ttmp;
        ttmp.unpack(receive_os);
        tmp_res[i]=compare_res_t[i]+ttmp;
    }

    for(int i=0;i<size_res;i++)
    {
        bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
        SignedZ2<K> dcf_u,dcf_v;
        dcf_res_u = evaluate(tmp_res[i], K,m_playerno);
        tmp_res[i] += 1LL<<(K-1);
        dcf_res_v = evaluate(tmp_res[i], K,m_playerno);
        auto size = dcf_res_u.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
        if(size < 0)
            dcf_u = -dcf_u;
        size = dcf_res_v.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
        if(size < 0)
            dcf_v = -dcf_v;
        if(tmp_res[i].get_bit(K-1)){
            r_tmp = dcf_v - dcf_u + m_playerno;
        }
        else{
            r_tmp = dcf_v - dcf_u;
        }
        compare_res[2*i]=SignedZ2<K>(m_playerno)-r_tmp;

        // compare_res[2*i]=evaluate(tmp_res[i],K,m_playerno);
        compare_res[2*i+1]=compare_res[2*i];//重复一次   
    }

}




void KNN_party_base::compare_in_vec(vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,vector<Z2<K>>&compare_res,bool greater_than)
{
    assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
    bigint r_tmp;
    fstream r;
    r.open("Player-Data/2-fss/r" + to_string( m_playerno), ios::in);
    r >> r_tmp;
    r.close();
    SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
    // cout<<"\n bigint:"<<r_tmp<<"  Z2<K>: " << alpha_share<<endl;
    int size_res=compare_idx_vec.size()/2;


    vector<SignedZ2<K>>compare_res_t(compare_res.size());
    if(greater_than)
    {
        for(int i=0;i<size_res;i++)
        {
            compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[2*i+1]][0])-SignedZ2<K>(shares[compare_idx_vec[2*i]][0])+alpha_share;
            //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
        }
    }
    else{
        for(int i=0;i<size_res;i++)
        {
            compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[2*i]][0])-SignedZ2<K>(shares[compare_idx_vec[2*i+1]][0])+alpha_share;
            //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
        }
    }
    
    // cout<<endl;
       
    vector<SignedZ2<K>>tmp_res(size_res);

    octetStream send_os,receive_os;
    for(int i=0;i<size_res;i++)compare_res_t[i].pack(send_os);
    player->send(send_os);
    player->receive(receive_os);
    for(int i=0;i<size_res;i++)
    {
        SignedZ2<K>ttmp;
        ttmp.unpack(receive_os);
        tmp_res[i]=compare_res_t[i]+ttmp;
    }

    for(int i=0;i<size_res;i++)
    {
        bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
        SignedZ2<K> dcf_u,dcf_v;
        dcf_res_u = evaluate(tmp_res[i], K,  m_playerno);
        tmp_res[i] += 1LL<<(K-1);
        dcf_res_v = evaluate(tmp_res[i], K, m_playerno);
        auto size = dcf_res_u.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
        if(size < 0)
            dcf_u = -dcf_u;
        size = dcf_res_v.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
        if(size < 0)
            dcf_v = -dcf_v;
        if(tmp_res[i].get_bit(K-1)){
            r_tmp = dcf_v - dcf_u + m_playerno;
        }
        else{
            r_tmp = dcf_v - dcf_u;
        }
        compare_res[2*i]=SignedZ2<K>(m_playerno)-r_tmp;

        // compare_res[2*i]=evaluate(tmp_res[i],K,m_playerno);
        compare_res[2*i+1]=compare_res[2*i];//重复一次   
    }

}


 

void KNN_party_optimized::top_1(vector<array<Z2<K>,2>>&shares,int size_now,bool min_in_last)
{
    
    std::vector<int> compare_idx_vec;
    for(int i=0;i<size_now;i++)compare_idx_vec.push_back(i);
    int leftover = -1; // 用于存储前一次迭代的剩余元素
    while (compare_idx_vec.size() + (leftover == -1 ? 0 : 1) >1 ) 
    {
        // 处理当前向量数组元素个数为奇数的情况
        if (compare_idx_vec.size() % 2 == 1) 
        {
            if (leftover == -1) 
            {
                // 如果没有前面的剩余元素，则取出最后一个元素
                leftover = compare_idx_vec.back();
                compare_idx_vec.pop_back();
            } 
            else 
            {
                // 否则将前面迭代的保留元素放入当前数组
                compare_idx_vec.push_back(leftover);
                leftover = -1; // 清空剩余元素
            }
        }

        vector<Z2<K>>compare_res(compare_idx_vec.size());
        compare_in_vec(shares,compare_idx_vec,compare_res,!min_in_last);
        SS_vec(shares,compare_idx_vec,compare_res);


        std::vector<int> new_compare_idx_vec;
            // 选择每隔一个元素
        for (size_t i = 1; i < compare_idx_vec.size(); i += 2) {
            new_compare_idx_vec.push_back(compare_idx_vec[i]);
        }

        // 将新向量赋值回原向量
        compare_idx_vec = std::move(new_compare_idx_vec);
    }

}


void KNN_party_base::SS_scalar(vector<array<Z2<K>,2>>&shares,int first_idx,int second_idx,bool min_then_max)
{
    Z2<K>u=secure_compare(shares[first_idx][0],shares[second_idx][0],min_then_max);
    vector<Z2<K>> Y(4);
    // mul_additive(u,shares[first_idx][0],y1);
    // mul_additive(u,shares[second_idx][0],y2);
    mul_vector_additive({shares[first_idx][0],shares[second_idx][0],shares[first_idx][1],shares[second_idx][1]},{u,u},Y,true);
    shares[first_idx][0]=shares[first_idx][0]-Y[0]+Y[1];
    shares[second_idx][0]=shares[second_idx][0]+Y[0]-Y[1];

    // mul_additive(u,shares[first_idx][1],y1);
    // mul_additive(u,shares[second_idx][1],y2);
    shares[first_idx][1]=shares[first_idx][1]-Y[2]+Y[3];
    shares[second_idx][1]=shares[second_idx][1]+Y[2]-Y[3];
}


void KNN_party_base::SS_scalar(vector<Z2<K>>&shares,int first_idx,int second_idx,bool min_then_max)
{
    Z2<K>u=secure_compare(shares[first_idx],shares[second_idx],min_then_max);
    vector<Z2<K>> Y(2);
    mul_vector_additive({shares[first_idx],shares[second_idx]},{u,u},Y,false);
    shares[first_idx]=shares[first_idx]-Y[0]+Y[1];
    shares[second_idx]=shares[second_idx]+Y[0]-Y[1];

}

        

void KNN_party_base::SS_vec( vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res)
{
    assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
    int size_of_cur_cmp=compare_idx_vec.size();
    vector<Z2<K>>tmp_ss;
    for(int i=0;i<size_of_cur_cmp;i++)tmp_ss.push_back(shares[compare_idx_vec[i]][0]);
    for(int i=0;i<size_of_cur_cmp;i++)tmp_ss.push_back(shares[compare_idx_vec[i]][1]);
    vector<Z2<K>>tmp_res(size_of_cur_cmp*2);//前半部分存储x乘法值，后半部分存储label的乘法值
    mul_vector_additive(tmp_ss,compare_res,tmp_res,true);
    for(int i=0;i<size_of_cur_cmp/2;i++)
    {
        tmp_ss[2*i]=tmp_ss[2*i]-tmp_res[2*i]+tmp_res[2*i+1]; //如果比较结果为x1>x2->1,此时为[x_min,x_max]
        tmp_ss[2*i+1]=tmp_ss[2*i+1] + tmp_res[2*i]-tmp_res[2*i+1];
    }
    for(int i=0;i<size_of_cur_cmp;i++)shares[compare_idx_vec[i]][0]=tmp_ss[i];

    for(int i=0;i<size_of_cur_cmp/2;i++)//tmp_ss存储shares中根据compare_idx_vec整合起来的值，tmp_res存储比较的结果,两个都是双倍比较结果长度
    {
        tmp_ss[2*i+size_of_cur_cmp]=tmp_ss[2*i+size_of_cur_cmp]-tmp_res[2*i+size_of_cur_cmp]+tmp_res[2*i+1+size_of_cur_cmp];
        tmp_ss[2*i+1+size_of_cur_cmp]=tmp_ss[2*i+1+size_of_cur_cmp] + tmp_res[2*i+size_of_cur_cmp]-tmp_res[2*i+1+size_of_cur_cmp];
    }
    for(int i=0;i<size_of_cur_cmp;i++)shares[compare_idx_vec[i]][1]=tmp_ss[i+size_of_cur_cmp];
}


void KNN_party_base::mul_additive(Z2<K>x1,Z2<K>x2,Z2<K>&res)
{
    Z2<K>a(0),b(0),c(0);
    octetStream send_os,receive_os;
    (x1-a).pack(send_os);
    (x2-b).pack(send_os);
    player->send(send_os);
    player->receive(receive_os);
    Z2<K>tmp_e,tmp_f;
    tmp_e.unpack(receive_os);
    tmp_f.unpack(receive_os);

    Z2<K>e=tmp_e+x1-a;
    Z2<K>f=tmp_f+x2-b;
    Z2<K>r=f*a+e*b+c;;
    if(player->my_num())
        r=r+e*f;
    res=r;
}

void KNN_party_base::mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res , bool double_res)
{
    if(double_res)
    {
        assert(v1.size()==v2.size()*2&&v1.size()==res.size());
        Z2<K>a(0),b(0),c(0);
        octetStream send_os,receive_os;
        int half_size=v2.size();
        for(int i=0;i<half_size;i++)
        {
            (v1[i]-a).pack(send_os);
            (v2[i]-b).pack(send_os);
        }
        for(int i=0;i<half_size;i++)
        {
            (v1[i+half_size]-a).pack(send_os);
            (v2[i]-b).pack(send_os);
        }
        player->send(send_os);
        player->receive(receive_os);
        vector<Z2<K>>tmp(v1.size()*2);
        for(int i=0;i<half_size;i++)
        {
            tmp[2*i].unpack(receive_os);
            tmp[2*i+1].unpack(receive_os);
            tmp[2*i]=tmp[2*i]+v1[i]-a;
            tmp[2*i+1]= tmp[2*i+1]+v2[i]-b;
        }
        for(int i=0;i<half_size;i++)
        {   
            Z2<K>e=tmp[2*i];
            Z2<K>f=tmp[2*i+1];
            Z2<K>r=f*a+e*b+c;;
            if(player->my_num())
                r=r+e*f;
            res[i]=r;
        }

        for(int i=0;i<half_size;i++)
        {
            tmp[2*i].unpack(receive_os);
            tmp[2*i+1].unpack(receive_os);
            tmp[2*i]=tmp[2*i]+v1[i+half_size]-a;
            tmp[2*i+1]= tmp[2*i+1]+v2[i]-b;
        }
        for(int i=0;i<half_size;i++)
        {   
            Z2<K>e=tmp[2*i];
            Z2<K>f=tmp[2*i+1];
            Z2<K>r=f*a+e*b+c;;
            if(player->my_num())
                r=r+e*f;
            res[i+half_size]=r;
        }


    }
    else{
        assert(v1.size()==v2.size());
        Z2<K>a(0),b(0),c(0);
        octetStream send_os,receive_os;
        for(int i=0;i<(int)v1.size();i++)
        {
            (v1[i]-a).pack(send_os);
            (v2[i]-b).pack(send_os);
        }
        player->send(send_os);
        player->receive(receive_os);
        vector<Z2<K>>tmp(v1.size()*2);
        for(int i=0;i<(int)v1.size();i++)
        {
            tmp[2*i].unpack(receive_os);
            tmp[2*i+1].unpack(receive_os);
            tmp[2*i]=tmp[2*i]+v1[i]-a;
            tmp[2*i+1]= tmp[2*i+1]+v2[i]-b;
        }
        for(int i=0;i<(int)v1.size();i++)
        {   
            Z2<K>e=tmp[2*i];
            Z2<K>f=tmp[2*i+1];
            Z2<K>r=f*a+e*b+c;;
            if(player->my_num())
                r=r+e*f;
            res[i]=r;
        }
    }
    
}

Z2<K> KNN_party_base::secure_compare(Z2<K>x1,Z2<K>x2,bool greater_than)//x1>x2-->1   x1<x2-->0   x1==x2-->0
{
    // cout<<x1<<" "<<x2<<endl;
    bigint r_tmp;
    fstream r;
    r.open("Player-Data/2-fss/r" + to_string(m_playerno), ios::in);
    r >> r_tmp;
    r.close();
    SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
    SignedZ2<K>revealed=SignedZ2<K>(x2)-SignedZ2<K>(x1)+alpha_share;
    if(greater_than==false){
        revealed=SignedZ2<K>(x1)-SignedZ2<K>(x2)+alpha_share;
    }
    
    octetStream send_os,receive_os;
    revealed.pack(send_os);
    player->send(send_os);
    player->receive(receive_os);
    SignedZ2<K>ttmp;
    ttmp.unpack(receive_os);
    revealed+=ttmp;

    bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
    SignedZ2<K> dcf_u,dcf_v;
    dcf_res_u = evaluate(revealed, K,m_playerno);
    revealed += 1LL<<(K-1);
    dcf_res_v = evaluate(revealed, K,m_playerno);
    auto size = dcf_res_u.get_mpz_t()->_mp_size;
    mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
    if(size < 0)
        dcf_u = -dcf_u;
    size = dcf_res_v.get_mpz_t()->_mp_size;
    mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
    if(size < 0)
        dcf_v = -dcf_v;
    if(revealed.get_bit(K-1)){
        r_tmp = dcf_v - dcf_u + m_playerno;
    }
    else{
        r_tmp = dcf_v - dcf_u;
    }
    SignedZ2<K>res=SignedZ2<K>(m_playerno)-r_tmp;
    // std::cout<<"revealed secure_compare result :"<<reveal_one_num_to(Z2<K>(res),0)<<std::endl;
    return Z2<K>(res);

}


void KNN_party_optimized::compute_ESD_for_one_query(int idx_of_test)
{
    if(int(m_ESD_vec.size())!=num_train_data)
        m_ESD_vec.resize(num_train_data);
    for(int i=0;i<num_train_data;i++)
    {
        m_ESD_vec[i][0]=compute_ESD_two_sample(i,idx_of_test);
        if(playerno==0){
            m_ESD_vec[i][1]=Z2<K>(m_sample[i]->label) -m_Train_Triples_1[i][0];
        }
        else{
            m_ESD_vec[i][1]=m_Train_Triples_1[i][0];
        }
    }  
}




void KNN_party_base::read_meta_and_P0_sample_P1_query()
{
    result=load_result_from_file("KmeansRes");
    
    // 清空旧的指针数据（如果有的话）
    m_sample.clear();
    m_test.clear();

    // 指针指向结构体中的样本
    if(playerno==0){
        for (auto& s : result.m_sample) {
            m_sample.push_back(&s); // 
        }
    }
   else{
        for (auto& t : result.m_test) {
            m_test.push_back(&t); //
        }
   }
    num_features=result.num_features;
    num_train_data=result.num_train_data;
    num_test_data=result.num_test_data;
    
}

void KNN_party_optimized::aby2_share_reveal(int x,bool sample_data) 
/*
    x：reveal的样本的索引
*/
{
    if(sample_data)
    {
        assert(x>=0&&x<num_train_data);
        if(playerno==0)
        {
            cout<<"Revealed the "<<x<< "th sample's feature:"<<endl;
            octetStream os;
            m_player->receive(os);
            for(int i=0;i<num_features;i++)
            {
                Z2<K>tmp;
                tmp.unpack(os);
                cout<<m_train_aby2_share_vec[x][i][0]-tmp-m_train_aby2_share_vec[x][i][1]<<" ";
            }
            cout<<endl;
        }
        else
        {
            octetStream os;
            for(int j=0;j<num_features;j++)
            {
                m_train_aby2_share_vec[x][j][1].pack(os);
            }
            m_player->send(os);
        }
    }
    else
    {
        assert(x>=0&&x<num_test_data);
        if(playerno==0)
        {
            cout<<"Revealed the "<<x<< "th test's feature:"<<endl;
            octetStream os;
            m_player->receive(os);
            for(int i=0;i<num_features;i++)
            {
                Z2<K>tmp;
                tmp.unpack(os);
                cout<<m_test_aby2_share_vec[x][i][0]-tmp-m_test_aby2_share_vec[x][i][1]<<" ";
            }
            cout<<endl;
        }
        else
        {
            octetStream os;
            for(int j=0;j<num_features;j++)
            {
                m_test_aby2_share_vec[x][j][1].pack(os);
            }
            m_player->send(os);
        }
    }
    
}



void KNN_party_base::additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec)
{
    assert(data_vec.size()!=0&& data_vec.size()==shares.size() );
    octetStream os;
    PRNG prng;
    prng.ReSeed();
    for(int i=0;i<(int)data_vec.size();i++)
    {
        shares[i].randomize(prng);
        shares[i].pack(os);
        shares[i] = data_vec[i]-shares[i];
    }
    m_player->send(os);
}

void KNN_party_base::additive_share_data_vec(vector<Z2<K>>&shares)
{
    octetStream os;
    m_player->receive(os);
    int size_of_share=shares.size();
    for(int i=0;i<size_of_share;i++)
        shares[i].unpack(os);
}


void parse_argv(int argc, const char** argv)
{
  opt.add(
          "5000", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Port number base to attempt to start connections from (default: 5000)", // Help description.
          "-pn", // Flag token.
          "--portnumbase" // Flag token.
  );
  opt.add(
          "", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "This m_player's number (required if not given before program name)", // Help description.
          "-p", // Flag token.
          "--m_player" // Flag token.
  );
  opt.add(
          "", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Port to listen on (default: port number base + m_player number)", // Help description.
          "-mp", // Flag token.
          "--my-port" // Flag token.
  );
  opt.add(
          "localhost", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Host where Server.x or party 0 is running to coordinate startup "
          "(default: localhost). "
          "Ignored if --ip-file-name is used.", // Help description.
          "-h", // Flag token.
          "--hostname" // Flag token.
  );
  opt.add(
          "", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Filename containing list of party ip addresses. Alternative to --hostname and running Server.x for startup coordination.", // Help description.
          "-ip", // Flag token.
          "--ip-file-name" // Flag token.
  );
  opt.parse(argc, argv);
  if (opt.isSet("-p"))
    opt.get("-p")->getInt(playerno);
  else
    sscanf(argv[1], "%d", &playerno);
}



void gen_fake_dcf(int beta, int n)
{
   // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = 16;
    PRNG prng;
    prng.InitSeed();
    fstream k0, k1, r0, r1, r2;
    k0.open("Player-Data/2-fss/k0", ios::out);
    k1.open("Player-Data/2-fss/k1", ios::out);
    r0.open("Player-Data/2-fss/r0", ios::out);
    r1.open("Player-Data/2-fss/r1", ios::out);
    r2.open("Player-Data/2-fss/r2", ios::out);
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

bigint evaluate(Z2<K> x, int n,int playerID)
{
    call_evaluate_time++;
    auto start = std::chrono::high_resolution_clock::now();

    fstream k_in;
    PRNG prng;
    int b = playerID, xi;
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
    int lambda_bytes = 16;
    k_in.open("Player-Data/2-fss/k" + to_string(playerID), ios::in);
    octet seed[lambda_bytes], tmp_seed[lambda_bytes];
    // r is the random value generate by GEN
    bigint s_hat[2], v_hat[2], t_hat[2], s[2], v[2], t[2], scw, vcw, tcw[2], convert[2], cw, tmp_t, tmp_v, tmp_out;
    k_in >> tmp_t;
    bytesFromBigint(&seed[0], tmp_t, lambda_bytes);
    tmp_t = b;
    tmp_v = 0;
    for(int i = 0; i < n - 1; i++){
        xi = x.get_bit(n - i - 1);
        bigintFromBytes(tmp_out, &seed[0], lambda_bytes);
        k_in >> scw >> vcw >> tcw[0] >> tcw[1];
        prng.SetSeed(seed);
        for(int j = 0; j < 2; j++){
            prng.get(t_hat[j], 1);
            prng.get(v_hat[j], n);
            prng.get(s_hat[j] ,n);
            s[j] = s_hat[j] ^ (tmp_t * scw);
            t[j] = t_hat[j] ^ (tmp_t * tcw[j]);
        }  
        bytesFromBigint(&tmp_seed[0], v_hat[0], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[0], n); 
        bytesFromBigint(&tmp_seed[0], v_hat[1], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[1], n);
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * vcw) + (1^b) * (convert[xi] + tmp_t * vcw);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
    }
    k_in >> cw;
    k_in.close();
    prng.SetSeed(seed);
    prng.get(convert[0], n);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * cw) + (1^b) * (convert[0] + tmp_t * cw);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // 将本次运行时间累加到全局变量中
    total_duration += duration;

    return tmp_v;  
}


void KNN_party_base::LabelCompute_test()
{
    vector<Z2<K>>test_k_neighbors={Z2<K>(1),Z2<K>(0),Z2<K>(2),Z2<K>(2),Z2<K>(2)};

    vector<additive_share>share_k_neighbors(5);//k=5

    octetStream os;
    
    if(playerno==0)
    {
        PRNG prng;
        prng.ReSeed();
        for(int i=0;i<(int)test_k_neighbors.size();i++)
        {
            share_k_neighbors[i].randomize(prng);
            share_k_neighbors[i].pack(os);
            share_k_neighbors[i] = test_k_neighbors[i]-share_k_neighbors[i];
        }
        m_player->send(os);
    }else{
        m_player->receive(os);
        for(int i=0;i<share_k_neighbors.size();i++){
            share_k_neighbors[i].unpack(os);
        }
    }
    
    for(int i=0;i<test_k_neighbors.size();i++){
        Z2<K>tmp=reveal_one_num_to(share_k_neighbors[i],1);
        if(playerno==1){
            std::cout<<tmp<<" == "<<test_k_neighbors[i]<<endl;
        }
    }

    
    vector<vector<Z2<K>>> cmp_2d_vec_res(k_const,vector<Z2<K>>(k_const));
    vector<int>cmp_vec_idx;
    for(int i=0;i<k_const;i++){
        for(int j=0;j<k_const;j++){
            cmp_vec_idx.push_back(i);
            cmp_vec_idx.push_back(j);
        }
    }
    vector<Z2<K>>cmp_res_vec(cmp_vec_idx.size());
    compare_in_vec(share_k_neighbors,cmp_vec_idx,cmp_res_vec,true);
    for(int i=0;i<cmp_res_vec.size()/2;i++){
        cmp_2d_vec_res[cmp_vec_idx[2*i]][cmp_vec_idx[2*i+1]]=Z2<K>(m_playerno)-cmp_res_vec[2*i];
    }

    vector<Z2<K>>v1(k_const*k_const),v2(k_const*k_const),res(k_const*k_const);
    for(int i=0;i<k_const;i++){
        for(int j=0;j<k_const;j++){
            v1[i*k_const+j]=cmp_2d_vec_res[i][j];
            v2[i*k_const+j]=cmp_2d_vec_res[j][i];
        }
    }
    mul_vector_additive(v1,v2,res,false);

    vector<array<Z2<K>,2>>label_list_count_array(k_const);
    for(int i=0;i<k_const;i++){
        Z2<K>tmp(0);
        for(int j=0;j<k_const;j++){
            tmp+=res[i*k_const+j];
        }
        label_list_count_array[i]={tmp,share_k_neighbors[i]};
    }

    cout<<"测试结果："<<endl;
    for(int i=0;i<k_const;i++){
        Z2<K>tmp_0=reveal_one_num_to(label_list_count_array[i][0],1);//出现频率
        Z2<K>tmp_1=reveal_one_num_to(label_list_count_array[i][1],1);//对应label
        if(playerno==1){
            std::cout<<tmp_0<<"  "<<tmp_1<<endl;
        }
    }

}


void test_Z2()
{
    long long x=65;
    long long y=999;
    // long long z=18446744073709543838;
    Z2<64>a(x);
    Z2<64>b(y);
    // cin>>b;
    cout<<a<<endl;
    cout<<b<<endl;
     std::vector<Z2<64>> tt(10);
    for (int i = 0; i < 10; i++) tt[i] = Z2<64>(i); // 仅初始化10个元素
    SignedZ2<K>c=a-b;
    cout<<SignedZ2<K>(a)<<endl;
    cout<<c<<endl;

    vector<Z2<K>>tmp1_vec(10);
    vector<Z2<K>>tmp2_vec(10);
    tmp1_vec[0]=Z2<64>(244);
    tmp2_vec[0]=Z2<64>(1);
    tmp1_vec[1]=Z2<64>(0);
    tmp2_vec[1]=Z2<64>(1);
    tmp1_vec[2]=Z2<64>(24);
    tmp2_vec[2]=Z2<64>(24);
    // KNN_party_base Kbase;
    cout<<endl;

}

void KNN_party_base::test_cmp()
{
    vector<Z2<K>>test_vec={Z2<K>(1),Z2<K>(23),Z2<K>(2),Z2<K>(2),Z2<K>(2)};
    vector<additive_share>share_test_k(5);//k=5
    octetStream os;
    if(playerno==0)
    {
        PRNG prng;
        prng.ReSeed();
        for(int i=0;i<(int)share_test_k.size();i++)
        {
            share_test_k[i].randomize(prng);
            share_test_k[i].pack(os);
            share_test_k[i] = test_vec[i]-share_test_k[i];
        }
        m_player->send(os);
    }else{
        m_player->receive(os);
        for(int i=0;i<share_test_k.size();i++){
            share_test_k[i].unpack(os);
        }
    }
    
    for(int i=0;i<test_vec.size();i++){
        Z2<K>tmp=reveal_one_num_to(share_test_k[i],1);
        if(playerno==1){
            std::cout<<tmp<<" == "<<test_vec[i]<<endl;
        }
    }
    Z2<K>res1=secure_compare(share_test_k[0],share_test_k[1],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
    Z2<K>res2=secure_compare(share_test_k[1],share_test_k[0],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
    Z2<K>res3=secure_compare(share_test_k[0],share_test_k[0],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
    cout<<"比较测试: x1 > x2-->1  else -->0"<<endl;
    Z2<K>tmp=reveal_one_num_to(res1,1);
    if(playerno==1){
        std::cout<<tmp<<" == "<<test_vec[0]<<">"<<test_vec[1]<<endl;
    }

    tmp=reveal_one_num_to(res2,1);
    if(playerno==1){
        std::cout<<tmp<<" == "<<test_vec[1]<<">"<<test_vec[0]<<endl;
    }

    tmp=reveal_one_num_to(res3,1);
    if(playerno==1){
        std::cout<<tmp<<" == "<<test_vec[0]<<">"<<test_vec[0]<<endl;
    }
    

}


