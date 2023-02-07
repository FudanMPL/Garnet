/*
 * Node.cpp
 *
 */

#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fstream>
#include <unistd.h>
#include <boost/thread.hpp>
#include "utils.h"
#include "Node.h"

static void throw_bad_map_file() {
	fprintf(stderr,"Node:: ERROR: could not read map file\n");
	throw std::invalid_argument( "bad map file" );
}

static void throw_bad_id(int id) {
	fprintf(stderr,"Node:: ERROR: bad id %d\n",id);
	throw std::invalid_argument( "bad id" );
}

Node::Node(const char* netmap_file, int my_id, NodeUpdatable* updatable, int num_parties)
	:_id(my_id),
	 _connected_to_servers(false),
	 _num_parties_identified(0),
	 _updatable(updatable)
{
	_parse_map(netmap_file, num_parties);
	unsigned int max_message_size = NETWORK_BUFFER_SIZE/2;
	if(_id < 0 || _id > _numparties)
		throw_bad_id(_id);
	_ready_nodes = new bool[_numparties](); //initialized to false
	_clients_connected = new bool[_numparties]();
	_server = new BIU::Server(_port, _numparties-1, this, max_message_size);
	_client = new BIU::Client(_endpoints, _numparties-1, this, max_message_size);
}

Node::~Node() {
	print_waiting();
	delete(_client);
	delete(_server);
	delete[] (_endpoints);
	delete[] (_ready_nodes);
	delete[] (_clients_connected);
}


void Node::Start() {
	_client->Connect();
	boost::thread(&Node::_start, this).join();
	_server->starter->join();
	for (unsigned int i = 0; i < _server->listeners.size(); i++)
		_server->listeners[i]->join();
}

void Node::Stop() {
	_client->Stop();
}

void Node::_start() {
	int interval = 10000;
	int total_wait = 0;
	while(true) {
		bool all_ready = true;
		if (_connected_to_servers) {
			for(int i=0; i<_numparties; i++) {
				if(i!=_id && _ready_nodes[i]==false) {
					all_ready = false;
					break;
				}
			}
		} else {
			all_ready = false;
		}
		if(all_ready)
			break;
		fprintf(stderr,"+");
//		fprintf(stderr,"Node:: waiting for all nodes to get ready ; sleeping for %u usecs\n", START_INTERVAL);
		if (interval < CONNECT_INTERVAL)
			interval *= 2;
		usleep(interval);
		total_wait += interval;
		if (total_wait > 60e6)
			throw runtime_error("waiting for too long");
	}
	printf("All identified\n");
	SendBuffer buffer;
	buffer.serialize(ALL_IDENTIFIED, strlen(ALL_IDENTIFIED));
	_client->Broadcast(buffer);
}

void Node::NewMsg(ReceivedMsg& message, struct sockaddr_in* from) {
	char* msg = message.data();
	size_t len = message.size();
#ifdef DEBUG_COMM
	printf("Node:: got message of length %u at 0x%x\n",len,msg);
	printf("from %s:%d\n", inet_ntoa(from->sin_addr), ntohs(from->sin_port));
	phex(msg, 4);
#endif	
	if(len == strlen(ID_HDR)+sizeof(_id) && 0==strncmp(msg, ID_HDR, strlen(ID_HDR))) {
		int *id = (int*)(msg+strlen(ID_HDR));
		printf("Node:: identified as party: %d\n", *id);
		assert(*id >=0 && *id <= _numparties && *id !=_id && !_clients_connected[*id]);
		_clients_connected[*id] = true;
		_clientsmap.insert(std::pair<struct sockaddr_in*,int>( from, *id));
		_ready_nodes[*id] = true;
		printf("Node:: _ready_nodes[%d]=%d\n",*id,_ready_nodes[*id]);
		return;
	} else if (len == strlen(ALL_IDENTIFIED) && 0==strncmp(msg, ALL_IDENTIFIED, strlen(ALL_IDENTIFIED))) {
		printf("Node:: received ALL_IDENTIFIED from %d\n",_clientsmap[from]);
		_num_parties_identified++;
		if(_num_parties_identified == _numparties-1) {
			printf("Node:: received ALL_IDENTIFIED from ALL\n");
			_updatable->NodeReady();
		}
		return;
	}
	_updatable->NewMessage(_clientsmap[from], message);
#ifdef DEBUG_COMM
	printf("finished with %d bytes at 0x%x\n", len, msg);
	phex(msg, 4);
#endif
}

void Node::ClientsConnected() {
	printf("Node:: Clients connected!\n");
}

void Node::NodeAborted(struct sockaddr_in* from)
{
	printf("Node:: party %d has aborted\n",_clientsmap[from]);
	Stop();
}

void Node::ConnectedToServers() {
	printf("Node:: Connected to all servers!\n");
	_connected_to_servers = true;
	_identify();
}

void Node::Send(int to, SendBuffer& msg) {
	int new_recipient = to>_id?to-1:to;
#ifdef DEBUG_COMM
	printf("Node:: new_recipient=%d\n",new_recipient);
	printf("Send %d bytes at 0x%x\n", msg.size(), msg.data());
	phex(msg.data(), 4);
#endif
	_client->Send(new_recipient, msg);
}

void Node::Broadcast(SendBuffer& msg) {
#ifdef DEBUG_COMM
	printf("Broadcast %d bytes at 0x%x\n", msg.size(), msg.data());
	phex(msg.data(), 4);
#endif
	_client->Broadcast(msg);
}
void Node::Broadcast2(SendBuffer& msg) {
#ifdef DEBUG_COMM
	printf("Broadcast2 %d bytes at 0x%x\n", msg.size(), msg.data());
	phex(msg.data(), 4);
#endif
	_client->Broadcast2(msg);
}

void Node::_identify() {
	char msg[strlen(ID_HDR)+sizeof(_id)];
	memcpy(msg, ID_HDR, strlen(ID_HDR));
	memcpy(msg+strlen(ID_HDR), (const char *)&_id, sizeof(_id));
	//printf("Node:: identifying myself:\n");
	SendBuffer buffer;
	buffer.serialize(msg, strlen(ID_HDR)+4);
#ifdef DEBUG_COMM
	cout << "message for identification:";
	phex(buffer.data(), 4);
#endif
	_client->Broadcast(buffer);
}

void Node::_parse_map(const char* netmap_file, int num_parties) {
	if(LOOPBACK == netmap_file) {
		_numparties = num_parties;
		_endpoints = new endpoint_t[_numparties-1];
		int j=0;
		for(int i=0; i<_numparties; i++) {
			if(_id == i) {
				_ip = LOCALHOST_IP;
				_port = PORT_BASE + i;
				//printf("Node:: my address: %s:%d\n", _ip.c_str(),_port);
				continue;
			}
			_endpoints[j].ip = LOCALHOST_IP;
			_endpoints[j].port = PORT_BASE + i;
			j++;
		}
	}
	else {
		std::ifstream netmap(netmap_file);
		if(!netmap.good()) throw_bad_map_file();

		netmap >> _numparties;
		_endpoints = new endpoint_t[_numparties-1];
		int j=0;
		for(int i=0; i<_numparties; i++) {
			if(_id == i) {
				netmap >> _ip >> _port;
#ifdef DEBUG_NETMAP
				printf("Node:: my address: %s:%d\n", _ip.c_str(),_port);
#endif
				continue;
			}
			netmap >> _endpoints[j].ip >> _endpoints[j].port;
#ifdef DEBUG_NETMAP
				printf("Node:: other address (%d): %s:%d\n", j,
						_endpoints[j].ip.c_str(), _endpoints[j].port);
#endif
			j++;
		}
	}
}

void Node::print_waiting()
{
	for (unsigned i = 0; i < _server->timers.size(); i++)
	{
		cout << "Waited " << _server->timers[i].elapsed()
				<< " seconds for client "
				<< _clientsmap[_server->get_client_addr(i)] << endl;
	}
}
