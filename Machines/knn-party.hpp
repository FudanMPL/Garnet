#include <iostream>
#include <fstream>


#include "../Machines/knn-party.h"
// #include <cmath>
using namespace std;

void KNN_party::start_networking(ez::ezOptionParser& opt) 
{
    cout<<3<<endl;
 
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

  std::cout<<4<<endl;
  Names playerNames;

  if (ipFileName.size() > 0) {
    if (my_port != Names::DEFAULT_PORT)
      throw runtime_error("cannot set port number when using IP file");
    playerNames.init(playerno, pnbase, ipFileName, nplayers);
  } else {
    Server::start_networking(playerNames, playerno, nplayers,
                             hostname, pnbase, my_port);
  }
  cout<<5<<endl;
  player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
}
