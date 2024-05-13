#include <iostream>
#include <random>
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
const int K=64;

 Z2<K>random_z2("10000000000");
int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;
void parse_argv(int argc, const char** argv);
void gen_fake_dcf(int beta, int n);
bigint evaluate(Z2<K> x, int n,int playerID);
void mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res);
void SS_vec( vector<Z2<K>>X , vector<Z2<K>>Y , vector<Z2<K>>&res);

class Sample{
public:
    vector<int> features;
    int label;
    Sample(int n):features(n){};
};

class KNN_party
{
public:
    typedef FixedVec<Z2<K>,2> aby2_share;
    TimerWithComm timer;
    int playerno = 0;//player编号
    const int nplayers = 2;
    int num_features;// 特征数
    int num_train_data; // 训练集数据总量
    int num_test_data; // 测试集数据总量

    RealTwoPartyPlayer* m_player; // 通信模块
    vector<Sample*>m_sample; //训练集
    vector<Sample*>m_test; //测试集

    array<PRNG, 2> shared_prngs;// 用不到这个了:   P0:拥有[0][1]    P1:拥有[0][2]，但是在这里P1用不到[2]
    vector<vector<aby2_share>>m_train_aby2_share_vec;
    vector<vector<aby2_share>>m_test_aby2_share_vec;

    vector< Z2<K> >m_ESD_vec;

    vector<vector< Z2<K> > >m_Train_Triples_0;  //P0 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> > >m_Train_Triples_1;  //P1 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> >>m_Test_Triples; // num_train_data * num_features  个三元组第三个值：[(\delta_x - \delta_y)*(\delta_x - \delta_y)]
    vector< Z2<K> > m_Test_Triples_P0;
    vector< Z2<K> > m_Test_Triples_P1;
    Z2<K> reveal_one_num_to(Z2<K> x,int playerID);
    void generate_triples_save_file();
    void load_triples();

    KNN_party(int playerNo):playerno(playerNo){};//构造函数
    void start_networking(ez::ezOptionParser& opt);//建立连接
    void rand_seed_set_up();
    void read_meta_and_P0_sample_P1_query();
    void aby2_share_data();
    void aby2_share_reveal(int idx,bool is_sample_data);

    void send_single_query(vector<Z2<K>> &query );
    int recv_single_answer();
    void share_data();
    void additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec);
    void additive_share_data_vec(vector<Z2<K>>&shares);
    void share_data_receive();
    Z2<K> compute_ESD_two_sample(int idx_of_sample,int idx_of_test);
    void compute_ESD_for_one_query(int idx_of_test);
    void compare_in_vec(vector<Z2<K>>&shares,vector<Z2<K>>&compare_res);

    vector<octetStream>data_send;
    vector<octetStream> data_receive;
    
    // vector< vector<  Z2<K>  > > triples( num_features, vector< Z2<K> >(3));
    
    void load_triples(string file_path);
   
   
    // void get_input_from(int palyerno,bool has_label,string file_path);//输入数据、、
    // vector<aby2> share_data_aby2();
    // vector<additive> compute_ESD(vector<aby2>&X,vector<aby2>&Y);
    void run();
};

int main(int argc, const char** argv)
{
    parse_argv(argc, argv);
    // test_Z2();

    KNN_party party(playerno);
    party.start_networking(opt);
    std::cout<<"Network Set Up Successful ! "<<std::endl;
    party.run();
    return 0;
}




void KNN_party::load_triples()
{
    if(playerno==0)
    {
        m_Train_Triples_0.resize(num_train_data);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        m_Test_Triples_P0.resize(num_features);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_0[i].resize(num_features);
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        ifstream file_Train_Triples_0("./Player-Data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/P0-Test-Triples", std::ios::binary),
            file_Train_Triples_1("./Player-Data/P1-Train-Triples", std::ios::binary);
        for(int i=0;i<num_features;i++)
        {
             file_Test_Triples_0.read(reinterpret_cast<char*>(&m_Test_Triples_P0[i]), sizeof(Z2<K>));
            //  cout<<m_Test_Triples_P0[i]<<" ";
        }
        // cout<<endl;
           
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
        cout<<"P0 loading triple ended!"<<endl;
    }
    else
    {
        m_Test_Triples_P0.resize(num_features);
        m_Test_Triples_P1.resize(num_features);
        m_Train_Triples_1.resize(num_train_data);
        m_Test_Triples.resize(num_train_data);
        for(int i=0;i<num_train_data;i++)
        {
            m_Train_Triples_1[i].resize(num_features);
            m_Test_Triples[i].resize(num_features);
        }
        ifstream file_Train_Triples_1("./Player-Data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/P1-Test-Triples", std::ios::binary);
        for(int i=0;i<num_features;i++)
        {
            file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_P0[i]), sizeof(Z2<K>));
            // cout<<m_Test_Triples_P0[i]<<" ";
        }
        // cout<<endl;
            
        for(int i=0;i<num_features;i++)
        {
            file_Test_Triples_1.read(reinterpret_cast<char*>(&m_Test_Triples_P1[i]), sizeof(Z2<K>));
            // cout<<m_Test_Triples_P1[i]<<" ";
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

        cout<<"P1 loading triple ended!"<<endl;
    }
    // vector<vector<Z2<K>>>Train_Triples_0(num_train_data,vector<Z2<K>>(num_features) );
    // vector<vector<Z2<K>>>Test_Triples_0(num_train_data+1,vector<Z2<K>>(num_features) );
    // vector<vector<Z2<K>>>Train_Triples_1(num_train_data,vector<Z2<K>>(num_features) );
    // vector<vector<Z2<K>>>Test_Triples_1(num_train_data+2,vector<Z2<K>>(num_features) );
    // ifstream file_Train_Triples_0("./Player-Data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/P0-Test-Triples", std::ios::binary),
    //         file_Train_Triples_1("./Player-Data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/P1-Test-Triples", std::ios::binary);
    // for(int i=0;i<num_features;i++)
    // {
    //     file_Test_Triples_0.read(reinterpret_cast<char*>(&Test_Triples_0[0][i]), sizeof(Z2<64>));
    //     file_Test_Triples_1.read(reinterpret_cast<char*>(&Test_Triples_1[0][i]), sizeof(Z2<64>));
    // }
    // for(int i=0;i<num_train_data;i++)
    // {
    //     for(int j=0;j<num_features;j++)
    //     {
    //         file_Train_Triples_0.read(reinterpret_cast<char*>(&Train_Triples_0[i][j]), sizeof(Z2<64>));
    //         file_Train_Triples_1.read(reinterpret_cast<char*>(&Train_Triples_1[i][j]), sizeof(Z2<64>));
    //         file_Test_Triples_0.read(reinterpret_cast<char*>(&Test_Triples_0[i+1][j]), sizeof(Z2<64>));
    //         file_Test_Triples_1.read(reinterpret_cast<char*>(&Test_Triples_1[i+1][j]), sizeof(Z2<64>));
    //     }
    // }
    // file_Train_Triples_0.close();
    // file_Train_Triples_1.close();
    // file_Test_Triples_0.close();
    // file_Test_Triples_1.close();
    //  for(int i=0;i<num_train_data;i++)
    // {
    //     for(int j=0;j<num_features;j++)
    //     {
    //         if(Test_Triples_0[i+1][j]+Test_Triples_1[i+1][j] == (Test_Triples_0[0][j]+Test_Triples_1[0][j]-Train_Triples_0[i][j]-Train_Triples_1[i][j])*
    //                                     (Test_Triples_0[0][j]+Test_Triples_1[0][j]-Train_Triples_0[i][j]-Train_Triples_1[i][j]))
    //                                     cout<<"== ";
    //         else cout<<"!! ";
    //     }
    //     cout<<endl;
    // }

}

void KNN_party::generate_triples_save_file()
{
    if(playerno==0)
    {
        PRNG seed;
        seed.ReSeed();
        vector<vector<Z2<K>>>Train_Triples_0(num_train_data,vector<Z2<K>>(num_features) );
        vector<vector<Z2<K>>>Test_Triples_0(num_train_data+1,vector<Z2<K>>(num_features) );
        vector<vector<Z2<K>>>Train_Triples_1(num_train_data,vector<Z2<K>>(num_features) );
        vector<vector<Z2<K>>>Test_Triples_1(num_train_data+1,vector<Z2<K>>(num_features) );
        ofstream file_Train_Triples_0("./Player-Data/P0-Train-Triples", std::ios::binary),file_Test_Triples_0("./Player-Data/P0-Test-Triples", std::ios::binary),
            file_Train_Triples_1("./Player-Data/P1-Train-Triples", std::ios::binary),file_Test_Triples_1("./Player-Data/P1-Test-Triples", std::ios::binary);
        
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

void KNN_party::start_networking(ez::ezOptionParser& opt) 
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

void KNN_party::send_single_query(vector<Z2<64>> &query) 
{
  data_send.resize(query.size());
  data_send[0].clear();
  int size = query.size();
  for (int i = 0; i < size; i++)
  {
    query[i].pack(data_send[0]);
    
  }
  m_player->send(data_send[0]);
}

int KNN_party::recv_single_answer() 
{
  octetStream os;
  m_player->receive(os);
  std::cout<<os.get_length()<<"   "<<os.get_total_length()<<std::endl;
  vector<Z2<64>>t(5);
   for(int i=0;i<5;i++)
   {
        t[i].unpack(os);
        std::cout<<*t[i].get()<<std::endl;
   }
  
  return 0;
}

Z2<K> KNN_party::compute_ESD_two_sample(int train_idx,int query_idx)
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

Z2<K> KNN_party::reveal_one_num_to(Z2<K> x,int playerID)
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

void KNN_party::run()
{
    read_meta_and_P0_sample_P1_query();
    std::cout<<"sample size:"<<m_sample.size()<<std::endl;
    std::cout<<"test size:"<<m_test.size()<<std::endl;
    
    // rand_seed_set_up();
    // generate_triples_save_file();//这个函数必须独立运行，不能和后续load_triple一起使用。
    // cout<<"\n generate_triples_save_file success!"<<endl;
    // load_triples();
    //  aby2_share_data();

    vector<Z2<K>>X,Y;
    for(int i=0;i<10;i++)
    {
        X.push_back(Z2<K>(std::rand() % 100));
        Y.push_back(Z2<K>(0));
    }
    vector<Z2<K>>compare_res(5);
    vector<Z2<K>>compare_res_align(10);




    if(playerno==0)
    {        
        timer.start(m_player->total_comm());

        vector<Z2<K>>res(4);

        // aby2_share_reveal(0,true);
        // aby2_share_reveal(0,false);

        // gen_fake_dcf(1,K);
        // compute_ESD_for_one_query(0);


        vector<Z2<K>>share_X(X.size());
        vector<Z2<K>>share_Y(Y.size());
        vector<Z2<K>>share_Z(X.size());

        additive_share_data_vec(share_X,X);
        // additive_share_data_vec(share_Y);
        for(int i=0;i<X.size();i++)
            std::cout<< reveal_one_num_to(share_X[i],0)<<" " ;
        std::cout<<endl;

        compare_in_vec(share_X,compare_res);
        for(int i=0;i<compare_res.size();i++)
            std::cout<< reveal_one_num_to(compare_res[i],0)<<" " ;
        std::cout<<endl;

        for(int i=0;i<5;i++){
            compare_res_align[2*i]=compare_res[i];
            compare_res_align[2*i+1]=compare_res[i];
        }
        mul_vector_additive(share_X,compare_res_align,share_Y);
        SS_vec(share_X,share_Y,share_Z);



        // cout<<endl;
        // for(int i=0;i<X.size();i++)
        //     cout<< reveal_one_num_to(share_Y[i],0)<<" " ;
        // cout<<endl;


        // mul_vector_additive(share_X,share_Y,share_Z);

        for(int i=0;i<X.size();i++)
            cout<< reveal_one_num_to(share_Z[i],0)<<" " ;
        cout<<endl;

        timer.stop(m_player->total_comm());
        std::cout << "Client total time = " << timer.elapsed() << " seconds" << std::endl;
        std::cout << "Client Data sent = " << timer.mb_sent() << " MB"<<std::endl;
    }
    else
    {
        timer.start(m_player->total_comm());

        // aby2_share_reveal(0,true);
        // aby2_share_reveal(0,false);
        // compute_ESD_for_one_query(0);
        // vector<Z2<K>>res(4);
        // res[0]=Z2<K>(evaluate(-5,K,1));
        // res[1]=Z2<K>(evaluate(5,K,1));
        // res[2]=Z2<K>(evaluate(3,K,1));
        // res[3]=Z2<K>(evaluate(0,K,1));
        // for(int i=0;i<4;i++)
        // {
        //     cout<< reveal_one_num_to(res[i],0)<<" " ;
        // }
        
        vector<Z2<K>>share_X(X.size());
        vector<Z2<K>>share_Y(Y.size());
        vector<Z2<K>>share_Z(X.size());

        additive_share_data_vec(share_X);
        
        // additive_share_data_vec(share_Y,Y);

        for(int i=0;i<X.size();i++)
            reveal_one_num_to(share_X[i],0);

         compare_in_vec(share_X,compare_res);

         for(int i=0;i<compare_res.size();i++)
            reveal_one_num_to(compare_res[i],0);
        
        for(int i=0;i<5;i++){
            compare_res_align[2*i]=compare_res[i];
            compare_res_align[2*i+1]=compare_res[i];
        }

        mul_vector_additive(share_X,compare_res_align,share_Y);

        SS_vec(share_X,share_Y,share_Z);



        


        // for(int i=0;i<X.size();i++)
        //     reveal_one_num_to(share_Y[i],0);

        // mul_vector_additive(share_X,share_Y,share_Z);

        

        for(int i=0;i<X.size();i++)
            reveal_one_num_to(share_Z[i],0);

        
        timer.stop(m_player->total_comm());
        std::cout << "Client total time = " << timer.elapsed() << " seconds" << std::endl;
        std::cout << "Client Data sent = " << timer.mb_sent() << " MB"<<std::endl;
    }

}

void KNN_party::compare_in_vec(vector<Z2<K>>&shares,vector<Z2<K>>&compare_res)
{
    bigint r_tmp;
    fstream r;
    r.open("Player-Data/2-fss/r" + to_string(playerno), ios::in);
    r >> r_tmp;
    r.close();
    Z2<K>alpha_share=(Z2<K>)r_tmp;
    // cout<<"\n bigint:"<<r_tmp<<"  Z2<K>: " << alpha_share<<endl;
    int size_res=shares.size()/2;
    for(int i=0;i<size_res;i++)
        compare_res[i]=shares[2*i+1]-shares[2*i]+alpha_share;
    vector<Z2<K>>tmp_res(size_res);

    octetStream send_os,receive_os;
    for(int i=0;i<size_res;i++)compare_res[i].pack(send_os);
    player->send(send_os);
    player->receive(receive_os);
    for(int i=0;i<size_res;i++)
    {
        Z2<K>ttmp;
        ttmp.unpack(receive_os);
        tmp_res[i]=compare_res[i]+ttmp;
    }

    for(int i=0;i<size_res;i++)compare_res[i]=evaluate(tmp_res[i],K,playerno);
}



void KNN_party::compute_ESD_for_one_query(int idx_of_test)
{
    m_ESD_vec.resize(num_train_data);
    for(int i=0;i<num_train_data;i++)
    {
        m_ESD_vec[i]=compute_ESD_two_sample(i,idx_of_test);
    }
}

void KNN_party::read_meta_and_P0_sample_P1_query()
{
    std::ifstream meta_file ("Player-Data/Knn-meta");
    meta_file >> num_features;// 特征数
    meta_file >> num_train_data;
    meta_file >> num_test_data;
    meta_file.close();
    if(playerno==0)
    {
        std::ifstream sample_file ("Player-Data/P0-0-X-Train");//暂时写死为P0
        for (int i = 0; i < num_train_data; i++)
        {
            Sample*sample_ptr=new Sample(num_features);
            for (int j = 0; j < num_features; j++)
            {
                sample_file>>sample_ptr->features[j];
            }
            m_sample.push_back(sample_ptr);
        }
        sample_file.close();

        std::ifstream label_file ("Player-Data/P0-0-Y-Train");//暂时写死为P0
        for (int i = 0; i < num_train_data; i++){
            label_file>>m_sample[i]->label;
        }
        label_file.close();
        cout<<"P0 read training file end!"<<endl;
    }
    else
    {
        std::ifstream test_file ("Player-Data/P1-0-X-Test");//暂时写死为P1
        for (int i = 0; i < num_test_data; i++)
        {
            Sample*test_ptr=new Sample(num_features);
            for (int j = 0; j < num_features; j++)
            {
                test_file>>test_ptr->features[j];
            }
            m_test.push_back(test_ptr);
        }
        test_file.close();

        std::ifstream label_file ("Player-Data/P1-0-Y-Test");//暂时写死为P1
        for (int i = 0; i < num_test_data; i++){
            label_file>>m_test[i]->label;
        }
        label_file.close();
        cout<<"P1 read testing(query) file end!"<<endl;
    }
    
}

void KNN_party::aby2_share_reveal(int x,bool sample_data) 
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

void KNN_party::aby2_share_data()
{
    m_train_aby2_share_vec.resize(num_train_data);
    for(int i=0;i<num_train_data;i++)
        m_train_aby2_share_vec[i].resize(num_features);
    
    m_test_aby2_share_vec.resize(num_test_data);
    for(int i=0;i<num_test_data;i++)
        m_test_aby2_share_vec[i].resize(num_features);

    if(playerno==0)
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
        cout<<"Train data aby2_share sending ended!"<<endl;

        os.clear();
        m_player->receive(os);
        for(int i=0;i<num_test_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {
                m_test_aby2_share_vec[i][j][1] = m_Test_Triples_P0[j];
                m_test_aby2_share_vec[i][j][0].unpack(os);
            }
        }
        cout<<"Test data aby2_share receiving ended!"<<endl;

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
        cout<<"Train data aby2_share receiving ended!"<<endl;

        os.clear();
        for(int i=0;i<num_test_data;i++)
        {
            for(int j=0;j<num_features;j++)
            {   
                m_test_aby2_share_vec[i][j][1]=m_Test_Triples_P1[j];
                m_test_aby2_share_vec[i][j][0]=Z2<K>(m_test[i]->features[j])+m_Test_Triples_P0[j]+m_Test_Triples_P1[j];
                m_test_aby2_share_vec[i][j][0].pack(os);
            }
        }
        m_player->send(os);
        cout<<"Test data aby2_share  sending ended!"<<endl;

    } 

}


void KNN_party::additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec)
{
    assert(data_vec.size()!=0&& data_vec.size()==shares.size() );
    octetStream os;
    PRNG prng;
    prng.ReSeed();
    for(int i=0;i<data_vec.size();i++)
    {
        shares[i].randomize(prng);
        shares[i].pack(os);
        shares[i] = data_vec[i]-shares[i];
    }
    player->send(os);
}

void KNN_party::additive_share_data_vec(vector<Z2<K>>&shares)
{
    octetStream os;
    player->receive(os);
    for(int i=0;i<shares.size();i++)
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

    std::ofstream f1("test_file_0", std::ios::binary);
    for (auto& element : tt) {
        // f1.write(reinterpret_cast<char*>(&element), sizeof(Z2<64>));
        f1<<(bigint)element<<" ";
    }
    f1.close();

    std::vector<Z2<K>> vec(10); // 只需要10个元素
    std::ifstream f2("test_file_0", std::ios::binary);
    bigint tmp;
    for (auto& element : vec) {
        // f2.read(reinterpret_cast<char*>(&element), sizeof(Z2<64>));
        f2>>tmp;
        element=tmp;
    }
    f2.close();

    for (int i = 0; i < 10; i++)
        std::cout << vec[i] << "   ";
    std::cout << std::endl;

    cout<<endl;



}

void KNN_party::rand_seed_set_up()
{
    if(playerno==0)//P0发送第一个随机种子给P1
    {
        shared_prngs[0].ReSeed();
        octetStream os;
        os.append(shared_prngs[0].get_seed(), SEED_SIZE);
        m_player->send(os);
    
    }
    else//P1接收P1发送的随机种子
    {
        octetStream os;
        m_player->receive(os);
        shared_prngs[0].SetSeed(os.get_data());
    }
    shared_prngs[1].ReSeed();
    // vector<Z2<K>> shares(4);
    // FixedVec<Z2<K>,4>s;
    // s[0].randomize(shared_prngs[0]);
    // s[1].randomize(shared_prngs[0]);
    // s[2].randomize(shared_prngs[0]);
    // s[3].randomize(shared_prngs[1]);
    // cout<<typeid(s).name()<<" "<<typeid(s[0]).name()<<endl;
    // cout<<"FixedVec<Z2<K>,4>s: "<<s[0]<<" "<<s[1]<<" "<<s[2]<<" "<<s[3]<<endl;
    // shares[0].randomize(shared_prngs[0]);
    // shares[1].randomize(shared_prngs[0]);
    // shares[2].randomize(shared_prngs[0]);
    // shares[3].randomize(shared_prngs[1]);
	// //此时P0，P1的shared_prngs[0]相同，shared_prngs[1]各自私有。
    // cout<<"shared_prngs:"<<*shares[0].get()<<"  "<<*shares[1].get()<<" "<<*shares[2].get()<<" "<<*shares[3].get()<<endl;
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
        keep = bigint(a >> n - i - 1).get_ui() & 1;
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
    return tmp_v;  
}


void mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res)
{
    assert(v1.size()==v2.size());
    Z2<K>a(0),b(0),c(0);
    octetStream send_os,receive_os;
    for(int i=0;i<v1.size();i++)
    {
        (v1[i]-a).pack(send_os);
        (v2[i]-b).pack(send_os);
    }
    player->send(send_os);
    player->receive(receive_os);
    vector<Z2<K>>tmp(v1.size()*2);
    for(int i=0;i<v1.size();i++)
    {
        tmp[2*i].unpack(receive_os);
        tmp[2*i+1].unpack(receive_os);
        tmp[2*i]=tmp[2*i]+v1[i]-a;
        tmp[2*i+1]= tmp[2*i+1]+v2[i]-b;
    }
    for(int i=0;i<v1.size();i++)
    {   
        Z2<K>e=tmp[2*i];
        Z2<K>f=tmp[2*i+1];
        Z2<K>r=f*a+e*b+c;;
        if(player->my_num())
            r=r+e*f;
        res[i]=r;
    }
}

void SS_vec( vector<Z2<K>>X , vector<Z2<K>>v2 , vector<Z2<K>>&res)
{
    int size_half=X.size()/2;
    for(int i=0;i<size_half;i++)
    {
        res[2*i]=X[2*i]-v2[2*i]+v2[2*i+1];
        res[2*i+1]=v2[2*i]+X[2*i+1]-v2[2*i+1];
    }
}