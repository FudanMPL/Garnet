/*
 * ServerSocket.h
 *
 */

#ifndef NETWORKING_SERVERSOCKET_H_
#define NETWORKING_SERVERSOCKET_H_

#include <map>
#include <set>
 #include <queue>
#include <string>
using namespace std;

#include <pthread.h>
#include <netinet/tcp.h>

#include "Tools/WaitQueue.h"
#include "Tools/Signal.h"

class ServerJob;

class ServerSocket
{
protected:
    int main_socket, portnum;
    map<string,int> clients;
    std::set<string> used;
    Signal data_signal;
    pthread_t thread;

    vector<ServerJob*> jobs;

    // disable copying
    ServerSocket(const ServerSocket& other);

    void process_connection(int socket, const string& client_id);

    virtual void process_client(const string&) {}

public:
    ServerSocket(int Portnum);
    virtual ~ServerSocket();

    virtual void init();

    virtual void accept_clients();

    void wait_for_client_id(int socket, struct sockaddr dest);

    // This depends on clients sending their id.
    // Has to be thread-safe.
    int get_connection_socket(const string& id);
};

/*
 * ServerSocket where clients do not send any identifiers upon connecting.
 */
class AnonymousServerSocket : public ServerSocket
{
private:
    // No. of accepted connections in this instance
    queue<string> client_connection_queue;

    void process_client(const string& client_id);

public:
    AnonymousServerSocket(int Portnum) :
        ServerSocket(Portnum) { };
    void init();

    // Get socket and id for the last client who connected
    int get_connection_socket(string& client_id);

    void remove_client(const string& client_id);
};

#endif /* NETWORKING_SERVERSOCKET_H_ */
