/*
 * Client.cpp
 *
 */

#include "Client.h"
#include "common.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <errno.h>

#include <boost/thread.hpp>


static void throw_bad_ip(const char* ip) {
	fprintf(stderr,"Client:: Error: inet_aton - not a valid address? %s\n", ip);
	throw std::invalid_argument( "bad ip" );
}

namespace BIU
{

Client::Client(endpoint_t* endpoints, int numservers, ClientUpdatable* updatable, unsigned int max_message_size)
	:_max_msg_sz(max_message_size),
	 _numservers(numservers),
	 _updatable(updatable)
	 {
	_sockets = new int[_numservers](); // 0 initialized
	_servers = new struct sockaddr_in[_numservers];
	_msg_queues = new WaitQueue< shared_ptr<SendBuffer> >[_numservers];

	memset(_servers, 0, sizeof(*_servers));

	for (int i=0; i<_numservers; i++) {
		_sockets[i] = socket(AF_INET, SOCK_STREAM, 0);
		if(-1 == _sockets[i])
			fprintf(stderr,"Client:: Error: socket: \n%s\n",strerror(errno));

		_servers[i].sin_family = AF_INET;
		_servers[i].sin_port = htons(endpoints[i].port);
		if(0 == inet_aton(endpoints[i].ip.c_str(), (in_addr*)&_servers[i].sin_addr))
			throw_bad_ip(endpoints[i].ip.c_str());
	}
}

Client::~Client() {
	Stop();
	for (int i=0; i<_numservers; i++)
		close(_sockets[i]);
	delete[] _sockets;
	delete[] _servers;
	delete[] _msg_queues;
#ifdef DEBUG_COMM
	printf("Client:: Client deleted\n");
#endif
}

void Client::Connect() {
	for (int i=0; i<_numservers; i++)
		threads.add_thread(new boost::thread(&Client::_send_thread, this, i));
	threads.add_thread(new boost::thread(&Client::_connect, this));
}

void Client::Stop() {
	for (int i=0; i<_numservers; i++)
		_msg_queues[i].stop();
	threads.join_all();
	for (int i=0; i<_numservers; i++)
		shutdown(_sockets[i], SHUT_RDWR);
#ifdef DEBUG_COMM
	printf("Stopped client\n");
#endif
}

void Client::_connect() {
	boost::thread_group tg;
	for(int i=0; i<_numservers; i++) {
		boost::thread* connector = new boost::thread(&Client::_connect_to_server, this, i);
		tg.add_thread(connector);
//		usleep(rand()%50000); // prevent too much collisions... TODO: remove
	}
	tg.join_all();
	_updatable->ConnectedToServers();
}

void Client::_connect_to_server(int i) {
	printf("Client:: connecting to server %d\n",i);
	char *ip;
	int port = ntohs(_servers[i].sin_port);
	ip = inet_ntoa(_servers[i].sin_addr);
	int error = 0;
	int interval = 10000;
	int total_wait = 0;
	while (true ) {
		error = connect(_sockets[i], (struct sockaddr *)&_servers[i], sizeof(struct sockaddr));
		if (interval < CONNECT_INTERVAL)
			interval *= 2;
		if(!error)
			break;
		if (errno == 111) {
			fprintf(stderr,".");
		} else {
			fprintf(stderr,"Client:: Error (%d): connect to %s:%d: \"%s\"\n",errno, ip,port,strerror(errno));
			fprintf(stderr,"Client:: socket %d sleeping for %u usecs\n",i, interval);
		}
		usleep(interval);
		total_wait += interval;
		if (total_wait > 60e6)
			throw runtime_error("waiting for too long");
	}

	printf("\nClient:: connected to %s:%d\n", ip,port);
	// Using the following disables the automatic buffer size (ipv4.tcp_wmem)
	// in favour of the core.wmem_max, which is worse.
	//setsockopt(_sockets[i], SOL_SOCKET, SO_SNDBUF, &NETWORK_BUFFER_SIZE, sizeof(NETWORK_BUFFER_SIZE));
}

void Client::Send(int id, SendBuffer& buffer) {
    {
#ifdef DEBUG_COMM
        printf ("Client:: queued %u bytes to %d\n", buffer.size(), id);
        phex(buffer.data(), 4);
#endif
    	SendBuffer* tmp = new SendBuffer;
    	*tmp = buffer;
    	shared_ptr<SendBuffer> new_msg(tmp);
        _msg_queues[id].push(new_msg);
    }
}

void Client::Broadcast(SendBuffer& buffer) {
#ifdef DEBUG_COMM
	printf ("Client:: queued %u bytes to broadcast\n", buffer.size());
	phex(buffer.data(), 4);
#endif
	SendBuffer* tmp = new SendBuffer;
	*tmp = buffer;
	shared_ptr<SendBuffer> new_msg(tmp);
	for(int i=0;i<_numservers; i++) {
        _msg_queues[i].push(new_msg);
	}
}

void Client::Broadcast2(SendBuffer& buffer) {
#ifdef DEBUG_COMM
	printf ("Client:: queued %u bytes to broadcast to all but the server\n", buffer.size());
	phex(buffer.data(), 4);
#endif
	SendBuffer* tmp = new SendBuffer;
	*tmp = buffer;
	shared_ptr<SendBuffer> new_msg(tmp);
	// first server is always the trusted party so we start with i=1
	for(int i=1;i<_numservers; i++) {
        _msg_queues[i].push(new_msg);
	}
}

void Client::_send_thread(int i) {
	shared_ptr<SendBuffer> msg;
	while(_msg_queues[i].pop_dont_stop(msg))
		_send_blocking(*msg, i);
#ifdef DEBUG_COMM
	printf("Shutting down sender thread %d\n", i);
#endif
}

void Client::_send_blocking(SendBuffer& msg, int id) {
#ifdef DEBUG_COMM
	printf ("Client:: sending %llu bytes at 0x%x to %d\n", msg.size(), msg.data(), id);
	fflush(0);
#ifdef DEBUG2
	phex(msg.data(), msg.size());
#else
	phex(msg.data(), 4);
#endif
#endif
	int cur_sent = 0;
	size_t len = msg.size();
	cur_sent = send(_sockets[id], &len, sizeof(len), 0);
	if(sizeof(len) == cur_sent) {
		unsigned int total_sent = 0;
		unsigned int remaining = 0;
		while(total_sent != msg.size()) {
			remaining = (msg.size()-total_sent)>_max_msg_sz ? _max_msg_sz : (msg.size()-total_sent);
			cur_sent = send(_sockets[id], msg.data()+total_sent, remaining, 0);
			//printf("Client:: msg.len=%u, remaining=%u, total_sent=%u, cur_sent = %d\n",msg.len, remaining, total_sent,cur_sent);
			if(cur_sent == -1) {
				fprintf(stderr,"Client:: Error: send msg failed: %s\n",strerror(errno));
				assert(cur_sent != -1);
			}
			total_sent += cur_sent;
		}
	} else if (-1 == cur_sent){
		fprintf(stderr,"Client:: Error: send header failed: %s\n",strerror(errno));
	}
#ifdef DEBUG_COMM
	printf ("Client:: sent %u bytes at 0x%x to %d\n", msg.size(), msg.data(), id);
	fflush(0);
	phex(msg.data(), 4);
	fflush(0);
#endif
}

}
