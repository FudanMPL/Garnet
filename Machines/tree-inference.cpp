

#include <iostream>
#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "../Tools/int.h"
#include "../Machines/TreeInferenceClient.h"
#include "../Machines/TreeInferenceServer.h"
#include "../Machines/TreeInferenceClient.hpp"
#include "../Machines/TreeInferenceServer.hpp"


using namespace std;

ez::ezOptionParser opt;
int playerno;
RealTwoPartyPlayer* player;

void parse_argv(int argc, const char** argv){

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
          "This player's number (required if not given before program name)", // Help description.
          "-p", // Flag token.
          "--player" // Flag token.
  );
  opt.add(
          "", // Default.
          0, // Required?
          1, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Port to listen on (default: port number base + player number)", // Help description.
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

//void start_networking(){
//  string hostname, ipFileName;
//  int pnbase;
//  int my_port;
//  opt.get("--portnumbase")->getInt(pnbase);
//  opt.get("--hostname")->getString(hostname);
//  opt.get("--ip-file-name")->getString(ipFileName);
//  ez::OptionGroup* mp_opt = opt.get("--my-port");
//  if (mp_opt->isSet)
//    mp_opt->getInt(my_port);
//  else
//    my_port = Names::DEFAULT_PORT;
//
//  Names playerNames;
//
//  if (ipFileName.size() > 0) {
//    if (my_port != Names::DEFAULT_PORT)
//      throw runtime_error("cannot set port number when using IP file");
//    playerNames.init(playerno, pnbase, ipFileName, nplayers);
//  } else {
//      Server::start_networking(playerNames, playerno, nplayers,
//                               hostname, pnbase, my_port);
//  }
//  player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
//}



int main(int argc, const char** argv){
   parse_argv(argc, argv);

   if (playerno == 0){
     TreeInferenceServer server;
     server.start_networking(opt);
     server.run();
   }
  else{
    TreeInferenceClient client;
    client.start_networking(opt);
    client.run();
  }
}
