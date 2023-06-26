#include "Processor/ExternalClients.h"
#include "Networking/ServerSocket.h"
#include <netinet/in.h>
#include <arpa/inet.h>
#include <thread>

ExternalClients::ExternalClients(int party_num):
   party_num(party_num),
   ctx(0)
{
}

ExternalClients::~ExternalClients() 
{
  // close client sockets
  for (auto it = external_client_sockets.begin();
    it != external_client_sockets.end(); it++)
  {
    delete it->second;
  }
  for (map<int,AnonymousServerSocket*>::iterator it = client_connection_servers.begin();
    it != client_connection_servers.end(); it++)
  {
    delete it->second;
  }
  if (ctx)
    delete ctx;
}

void ExternalClients::start_listening(int portnum_base)
{
  client_connection_servers[portnum_base] = new AnonymousServerSocket(portnum_base + get_party_num());
  client_connection_servers[portnum_base]->init();
  cerr << "Start listening on thread " << this_thread::get_id() << endl;
  cerr << "Party " << get_party_num() << " is listening on port " << (portnum_base + get_party_num())
        << " for external client connections." << endl;
}

int ExternalClients::get_client_connection(int portnum_base)
{
  map<int,AnonymousServerSocket*>::iterator it = client_connection_servers.find(portnum_base);
  if (it == client_connection_servers.end())
  {
    cerr << "Thread " << this_thread::get_id() << " didn't find server." << endl; 
    throw runtime_error("No connection on port " + to_string(portnum_base));
  }
  cerr << "Thread " << this_thread::get_id() << " found server." << endl; 
  int client_id, socket;
  string client;
  socket = client_connection_servers[portnum_base]->get_connection_socket(
      client);
  client_id = stoi(client);
  if (ctx == 0)
    ctx = new client_ctx("P" + to_string(get_party_num()));
  external_client_sockets[client_id] = new client_socket(io_service, *ctx, socket,
      "C" + to_string(client_id), "P" + to_string(get_party_num()), false);
  client_ports[client_id] = portnum_base;
  cerr << "Party " << get_party_num() << " received external client connection from client id: " << dec << client_id << endl;
  return client_id;
}

void ExternalClients::close_connection(int client_id)
{
  auto it = external_client_sockets.find(client_id);
  if (it == external_client_sockets.end())
    throw runtime_error("client id not active: " + to_string(client_id));
  delete external_client_sockets[client_id];
  external_client_sockets.erase(client_id);
  client_connection_servers[client_ports[client_id]]->remove_client(
      to_string(client_id));
}

int ExternalClients::get_party_num() 
{
  return party_num;
}

client_socket* ExternalClients::get_socket(int id)
{
  if (external_client_sockets.find(id) == external_client_sockets.end())
    throw runtime_error("external connection not found for id " + to_string(id));
  return external_client_sockets[id];
}
