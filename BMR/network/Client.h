/*
 * Client.h
 *
 */

#ifndef NETWORK_INC_CLIENT_H_
#define NETWORK_INC_CLIENT_H_

#include "common.h"
//#include <queue>
//#include <boost/thread.hpp>
//#include <boost/thread/mutex.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/thread.hpp>

#include "Tools/WaitQueue.h"
#include "Tools/FlexBuffer.h"

#define CONNECT_INTERVAL (1000000)

class ClientUpdatable {
public:
	virtual void ConnectedToServers()=0;
};



namespace BIU
{

class Client {
public:
	Client(endpoint_t* servers, int numservers, ClientUpdatable* updatable, unsigned int max_message_size);
	virtual ~Client();

	void Connect();
	void Send(int id, SendBuffer& new_msg);
	void Broadcast(SendBuffer& new_msg);
	void Broadcast2(SendBuffer& new_msg);
	void Stop();

private:

	WaitQueue< shared_ptr<SendBuffer> >* _msg_queues;
//	std::queue<Msg>* _msg_queues;
//	boost::mutex* _msg_mux;
//	boost::mutex* _thd_mux;
	unsigned int _max_msg_sz;
	void _send_thread(int i);
	void _send_blocking(SendBuffer& msg, int id);

	void _connect();
	void _connect_to_server(int i);

	int _numservers;
	struct sockaddr_in* _servers;
	int* _sockets;
	ClientUpdatable* _updatable;

	boost::thread_group threads;
};

}

#endif /* NETWORK_INC_CLIENT_H_ */
