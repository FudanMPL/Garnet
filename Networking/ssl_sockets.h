/*
 * sockets.h
 *
 */

#ifndef CRYPTO_SSL_SOCKETS_H_
#define CRYPTO_SSL_SOCKETS_H_

#include "Tools/int.h"
#include "Tools/time-func.h"
#include "sockets.h"
#include "Math/Setup.h"

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

#ifndef SSL_DIR
#define SSL_DIR "Player-Data/"
#endif

typedef boost::asio::io_service ssl_service;

void check_ssl_file(string filename);
void ssl_error(string side, string other, string server);

class ssl_ctx : public boost::asio::ssl::context
{
public:
    ssl_ctx(string me) :
            boost::asio::ssl::context(boost::asio::ssl::context::tlsv12)
    {
        string prefix = SSL_DIR + me;
        string cert_file = prefix + ".pem";
        string key_file = prefix + ".key";
        check_ssl_file(cert_file);
        check_ssl_file(key_file);

        use_certificate_file(cert_file, pem);
        use_private_key_file(key_file, pem);
        add_verify_path(SSL_DIR);
    }
};

class ssl_socket : public boost::asio::ssl::stream<boost::asio::ip::tcp::socket>
{
    typedef boost::asio::ssl::stream<boost::asio::ip::tcp::socket> parent;

public:
    ssl_socket(boost::asio::io_service& io_service,
            boost::asio::ssl::context& ctx, int plaintext_socket, string other,
            string me, bool client) :
            parent(io_service, ctx)
    {
#ifdef DEBUG_NETWORKING
        cerr << me << " setting up SSL to " << other << " as " <<
                (client ? "client" : "server") << endl;
#endif
        lowest_layer().assign(boost::asio::ip::tcp::v4(), plaintext_socket);
        set_verify_mode(boost::asio::ssl::verify_peer);
        set_verify_callback(boost::asio::ssl::rfc2818_verification(other));
        if (client)
            try
            {
                handshake(ssl_socket::client);
            } catch (...)
            {
                ssl_error("Client", other, me);
                throw;
            }
        else
        {
            try
            {
                handshake(ssl_socket::server);
            } catch (...)
            {
                ssl_error("Server", other, me);
                throw;
            }

        }
    }
};

inline size_t send_non_blocking(ssl_socket* socket, octet* data, size_t length)
{
    return socket->write_some(boost::asio::buffer(data, length));
}

inline void send(ssl_socket* socket, octet* data, size_t length)
{
    size_t sent = 0;
#ifdef VERBOSE_SSL
    RunningTimer timer;
#endif
    while (sent < length)
    {
        sent += send_non_blocking(socket, data + sent, length - sent);
#ifdef VERBOSE_SSL
        cout << "sent " << sent * 1e-6 << " MB at " << timer.elapsed() << endl;
#endif
    }
}

inline void receive(ssl_socket* socket, octet* data, size_t length)
{
    size_t received = 0;
    while (received < length)
        received += socket->read_some(boost::asio::buffer(data + received, length - received));
}

inline size_t receive_non_blocking(ssl_socket* socket, octet* data, int length)
{
    return socket->read_some(boost::asio::buffer(data, length));
}

inline size_t receive_all_or_nothing(ssl_socket* socket, octet* data, size_t length)
{
    receive(socket, data, length);
    return length;
}

#endif /* CRYPTO_SSL_SOCKETS_H_ */
