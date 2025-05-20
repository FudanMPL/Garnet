
#include "Networking/sockets.h"
#include "Networking/ServerSocket.h"
#include "Networking/Server.h"

#include <iostream>
#include <pthread.h>
#include <assert.h>


/*
 * Get the client ip number on the socket connection for client i.
 */
void Server::get_ip(int num)
{
  struct sockaddr_storage addr;
  socklen_t len = sizeof addr;  

  getpeername(socket_num[num], (struct sockaddr*)&addr, &len);

  // supports both IPv4 and IPv6:
  char ipstr[INET6_ADDRSTRLEN];  
  if (addr.ss_family == AF_INET) {
      struct sockaddr_in *s = (struct sockaddr_in *)&addr;
      inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);
  } else { // AF_INET6
      struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
      inet_ntop(AF_INET6, &s->sin6_addr, ipstr, sizeof ipstr);
  }

  names[num] = ipstr;

#ifdef DEBUG_NETWORKING
  cerr << "Client IP address: " << names[num] << endl;
#endif
}


void Server::get_name(int num)
{
#ifdef DEBUG_NETWORKING
  cerr << "Player " << num << " started." << endl;
#endif

  // Receive name sent by client (legacy) - not used here
  octetStream os;
  os.Receive(socket_num[num]);
  receive(socket_num[num],(octet*)&ports[num],4);
#ifdef DEBUG_NETWORKING
  cerr << "Player " << num << " sent (IP for info only) " << os.str() << ":"
      << ports[num] << endl;
#endif

  // Get client IP
  get_ip(num);
}


void Server::send_names()
{
  /* Now send the machine names back to each client 
   * and the number of machines
   */
  RunningTimer timer;
  octetStream addresses;
  addresses.store(names);
  addresses.store(ports);
  for (int i=0; i<nmachines; i++)
    {
      addresses.Send(socket_num[i]);
    }
}


/* Takes command line arguments of 
       - Number of machines connecting
       - Base PORTNUM address
*/

Server::Server(int argc,char **argv)
{
  if (argc != 3)
    { cerr << "Call using\n\t";
      cerr << "Server.x n PortnumBase\n";
      cerr << "\t\t n           = Number of machines" << endl;
      cerr << "\t\t PortnumBase = Base Portnum\n";
      exit(1);
    }
  nmachines=atoi(argv[1]);
  PortnumBase=atoi(argv[2]);
  server_socket = 0;
}

Server::Server(int nmachines, int PortnumBase) :
    nmachines(nmachines), PortnumBase(PortnumBase), server_socket(0)
{
}

Server::~Server()
{
  if (server_socket)
    delete server_socket;
}

void Server::start()
{
  int i;

  names.resize(nmachines);
  ports.resize(nmachines);

  /* Set up the sockets */
  socket_num.resize(nmachines);
  for (i=0; i<nmachines; i++) { socket_num[i]=-1; }

  // port number one lower to avoid conflict with players
  server_socket = new ServerSocket(PortnumBase);
  auto& server = *server_socket;
  server.init();

  // set up connections
  for (i=0; i<nmachines; i++)
    {
#ifdef DEBUG_NETWORKING
      cerr << "Waiting for player " << i << endl;
#endif
      socket_num[i] = server.get_connection_socket("P" + to_string(i));
#ifdef DEBUG_NETWORKING
      cerr << "Connected to player " << i << endl;
#endif
    }

  // get names
  for (i=0; i<nmachines; i++)  
    get_name(i);  

  // check setup, party 0 doesn't matter
  bool all_on_local = true, none_on_local = true;
  for (i = 1; i < nmachines; i++)
    {
      bool on_local = names[i].compare("127.0.0.1");
      all_on_local &= on_local;
      none_on_local &= not on_local;
    }
  if (not all_on_local and not none_on_local)
    {
      cout << "You cannot address Server.x by localhost if using different hosts" << endl;
      exit(1);
    }

  // send names
  send_names();

  for (int i = 0; i < nmachines; i++)
    close(socket_num[i]);
}

void* Server::start_in_thread(void* server)
{
  ((Server*)server)->start();
  return 0;
}

Server* Server::start_networking(Names& N, int my_num, int nplayers,
        string hostname, int portnum, int my_port)
{
#ifdef DEBUG_NETWORKING
  cerr << "Starting networking for " << my_num << "/" << nplayers
      << " with server on " << hostname << ":" << (portnum) << endl;
#endif
  assert(my_num >= 0);
  assert(my_num < nplayers);
  Server* server = 0;
  pthread_t thread;
  if (my_num == 0)
    {
      pthread_create(&thread, 0, Server::start_in_thread,
          server = new Server(nplayers, portnum));
      bool default_port = my_port == Names::DEFAULT_PORT or my_port == portnum;
      N.init(my_num, portnum, my_port, hostname.c_str(), not default_port);
      pthread_join(thread, 0);
      if (default_port)
        N.set_server(server->get_socket());
      delete server;
    }
  else
      N.init(my_num, portnum, my_port, hostname.c_str());
  return 0;
}

ServerSocket* Server::get_socket()
{
  auto res = server_socket;
  server_socket = 0;
  return res;
}
