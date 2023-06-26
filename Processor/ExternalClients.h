#ifndef _ExternalClients
#define _ExternalClients

#include "Networking/sockets.h"
#include "Networking/ssl_sockets.h"
#include "Tools/Exceptions.h"
#include "ExternalIO/Client.h"
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sodium.h>
#include <assert.h>

class AnonymousServerSocket;

/*
 * Manage the reading and writing of data from/to external clients via Sockets.
 * Generate the session keys for encryption/decryption of secret communication with external clients.
 */

class ExternalClients
{
  map<int,AnonymousServerSocket*> client_connection_servers;
  
  int party_num;

  // Maps holding per client values (indexed by unique 32-bit id)
  std::map<int,client_socket*> external_client_sockets;
  std::map<int, int> client_ports;

  ssl_service io_service;
  client_ctx* ctx;

  public:

  ExternalClients(int party_num);
  ~ExternalClients();

  void start_listening(int portnum_base);

  int get_client_connection(int portnum_base);

  void close_connection(int client_id);

  // return the socket for a given client or server identifier
  client_socket* get_socket(int socket_id);

  int get_party_num();
};

#endif
