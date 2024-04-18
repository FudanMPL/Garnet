/*
 * Server.h
 *
 */

#ifndef NETWORK_INC_SERVER_H_
#define NETWORK_INC_SERVER_H_

#include <sys/socket.h>
#include <netinet/in.h>
#include <vector>

#include "common.h"
#include "Tools/FlexBuffer.h"

class ServerUpdatable {
public:
	virtual void ClientsConnected()=0;
	virtual void NewMsg(ReceivedMsg& msg, struct sockaddr_in* from)=0;
	virtual void NodeAborted(struct sockaddr_in* from) =0;
};

namespace BIU
{

class Server {
public:
	Server(int port, int expected_clients, ServerUpdatable* updatable, unsigned int max_message_size);
	~Server();

	sockaddr_in* get_client_addr(int id) { return &_clients_addr[id]; }

	boost::thread* starter;
	std::vector<boost::thread*> listeners;

	vector<Timer> timers;

private:
	int _expected_clients;
	int *_clients;
	struct sockaddr_in* _clients_addr;
	int _port;
	int _servfd;
	struct sockaddr_in _servaddr;

	ServerUpdatable* _updatable;
	unsigned int _max_msg_sz;

	void _start_server();
	void _listen_to_client(int id);
	bool _handle_recv_len(int id, size_t actual_len, size_t expected_len);
};

}

#endif /* NETWORK_INC_SERVER_H_ */
