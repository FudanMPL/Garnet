#include <iostream>
#include <fstream>
#include "../Machines/knn-party.h"
// using namespace std;
typedef unsigned long size_t;


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
    this->player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
  }

void KNN_party::send_single_query(vector<Z2<64>> &query) 
{
  data_send.resize(query.size());
  int size = query.size();
  for (int i = 0; i < size; i++)
  {
    query[i].pack(data_send[0]);
    
  }
  player->send(data_send[0]);
  // cout<<*query[0].get()<<endl;
  // player->send(data_send);
}


int KNN_party::recv_single_answer() 
{
  octetStream os;
  player->receive(os);
  std::cout<<os.get_length()<<"   "<<os.get_total_length()<<std::endl;
  vector<Z2<64>>t(5);
   for(int i=0;i<5;i++)
   {
      t[i].unpack(os);
      std::cout<<*t[i].get()<<std::endl;
   }
  
  return 0;
}


void KNN_party::run()
{
  // read_meta_and_sample();

  if(playerno==0)
  {
    timer.start(player->total_comm());
    vector< Z2<64> >tmp;
    for(int i=0;i<5;i++)tmp.push_back(i+10);
     std::cout<<"In: "<<playerno<<std::endl;
    send_single_query(tmp);
    std::cout<<"Sender send success! "<<std::endl;

    timer.stop(player->total_comm());
    std::cout << "Client total time = " << timer.elapsed() << " seconds" << std::endl;
    std::cout << "Client Data sent = " << timer.mb_sent() << " MB";
  }
  else
  {
    timer.start(player->total_comm());
    recv_single_answer();
    std::cout<<"Sender send success! "<<std::endl;
    timer.stop(player->total_comm());
    std::cout << "Client total time = " << timer.elapsed() << " seconds" << std::endl;
    std::cout << "Client Data sent = " << timer.mb_sent() << " MB";
  }
}