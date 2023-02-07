/*
 * Exchanger.h
 *
 */

#ifndef NETWORKING_EXCHANGER_H_
#define NETWORKING_EXCHANGER_H_

#include "Tools/octetStream.h"
#include "Tools/time-func.h"
#include "sockets.h"

template<class T>
class Exchanger
{
    T send_socket, receive_socket;
    const octetStream& send_stream;
    octetStream& receive_stream;

    size_t sent, received;
    bool length_received;
    size_t new_len;
    size_t n_iter, n_send;

    size_t len;
    octet* data;

    Timer recv_timer;
    Timer send_timer;

public:
    Exchanger(T send_socket, const octetStream& send_stream, T receive_socket,
            octetStream& receive_stream) :
            send_socket(send_socket), receive_socket(receive_socket), send_stream(
                    send_stream), receive_stream(receive_stream)
    {
        len = send_stream.get_length();
        data = send_stream.get_data();
        send(send_socket, len, LENGTH_SIZE);
        sent = 0;
        received = 0;
        length_received = false;
        new_len = 0;
        n_iter = 0;
        n_send = 0;
    }

    ~Exchanger()
    {
#ifdef TIME_ROUNDS
  cout << "Exchange time: " << recv_timer.elapsed() << " seconds and " << n_iter <<
          " iterations to receive "
      << 1e-3 * new_len << " KB, " << send_timer.elapsed() << " seconds and " << n_send
      << " iterations to send " << 1e-3 * len << " KB" << endl;
#endif
        receive_stream.len = new_len;
        receive_stream.reset_read_head();
    }

    bool round(bool block = true)
    {
        n_iter++;
        if (sent < len)
        {
#ifdef TIME_ROUNDS
                TimeScope ts(send_timer);
      #endif
            n_send++;
            size_t to_send = len - sent;
            size_t newly_sent = send_non_blocking(send_socket, data + sent,
                    to_send);
#ifdef TIME_ROUNDS
                cout << "sent " << newly_sent << "/" << to_send << endl;
      #endif
            sent += newly_sent;
        }

        // avoid extra branching, false before length received
        if (received < new_len)
        {
            // if same buffer for sending and receiving
            // only receive up to already sent data
            // or when all is sent
            size_t to_receive = 0;
            if (sent == len or &send_stream != &receive_stream)
                to_receive = new_len - received;
            else if (sent > received)
                to_receive = sent - received;
            if (to_receive > 0)
            {
#ifdef TIME_ROUNDS
                    TimeScope ts(recv_timer);
      #endif
                if (sent < len or not block)
                {
                    size_t newly_received = receive_non_blocking(receive_socket,
                            receive_stream.data + received, to_receive);
#ifdef TIME_ROUNDS
                        cout << "received " << newly_received << "/" << to_receive << endl;
      #endif
                    received += newly_received;
                }
                else
                {
                    receive(receive_socket, receive_stream.data + received,
                            to_receive);
                    received += to_receive;
                }
            }
        }
        else if (not length_received)
        {
#ifdef TIME_ROUNDS
                TimeScope ts(recv_timer);
      #endif
            octet blen[LENGTH_SIZE];
            size_t tmp = LENGTH_SIZE;
            if (sent < len or not block)
                tmp = receive_all_or_nothing(receive_socket, blen, LENGTH_SIZE);
            else
                receive(receive_socket, blen, LENGTH_SIZE);
            if (tmp == LENGTH_SIZE)
            {
                new_len = decode_length(blen, sizeof(blen));
                receive_stream.resize(max(new_len, len));
                length_received = true;
            }
        }

        return (received < new_len or sent < len or not length_received);
    }
};

#endif /* NETWORKING_EXCHANGER_H_ */
