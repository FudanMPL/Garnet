/*
 * Server.h
 */

#ifndef NETWORKING_SERVER_H_
#define NETWORKING_SERVER_H_

#include "Networking/data.h"
#include "Networking/Player.h"

#include <vector>
using namespace std;

class Server
{
    vector<int> socket_num;
    vector<string> names;
    vector<int> ports;
    int nmachines;
    int PortnumBase;
    ServerSocket* server_socket;

    void get_ip(int num);
    void get_name(int num);
    void send_names();

public:
    static void* start_in_thread(void* server);
    static Server* start_networking(Names& N, int my_num, int nplayers,
            string hostname = "localhost", int portnum = 9000, int my_port =
                    Names::DEFAULT_PORT);

    Server(int argc, char** argv);
    Server(int nmachines, int PortnumBase);
    ~Server();

    void start();

    ServerSocket* get_socket();
};

#endif /* NETWORKING_SERVER_H_ */
