#include <iostream>
#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "../Tools/int.h"
#include "../Math/bigint.h"
#include "../Math/Z2k.h"

#include "../Machines/knn-party.h"
#include "../Machines/knn-party.hpp"


using namespace std;
void test_Z2();
const int K=64;

ez::ezOptionParser opt;
int playerno;
RealTwoPartyPlayer* player;

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




int main(int argc, const char** argv)
{
        parse_argv(argc, argv);
    cout<<1<<endl;
    KNN_party party;
    cout<<2<<endl;
    party.start_networking(opt);
    // party.run();
    return 0;
}

void test_Z2()
{
    long long x=65;
    long long y=-7778;
    // long long z=18446744073709543838;
    Z2<128>a(x);
    Z2<128>b(y);
    Z2<128>c("18446744073709543838");
    cout<<a.size_in_limbs()<<"  "<<a.size_in_bits()<<endl;
    cout<<*b.get()<<endl;
    cout<<*(a*b).get()<<endl;
    cout<<*c.get()<<endl;
    cout<<*(a*c).get()<<endl;
}

