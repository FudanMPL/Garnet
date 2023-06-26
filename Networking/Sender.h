/*
 * Sender.h
 *
 */

#ifndef NETWORKING_SENDER_H_
#define NETWORKING_SENDER_H_

#include <pthread.h>

#include "Tools/octetStream.h"
#include "Tools/WaitQueue.h"
#include "Tools/time-func.h"

template<class T>
class Sender
{
    T socket;
    WaitQueue<const octetStream*> in;
    WaitQueue<const octetStream*> out;
    pthread_t thread;

    static void* run_thread(void* sender);

    // prevent copying
    Sender(const Sender& other);

    void start();
    void stop();
    void run();

public:
    Timer timer;

    Sender(T socket);
    ~Sender();

    T get_socket()
    {
        return socket;
    }

    void request(const octetStream& os);
    void wait(const octetStream& os);
};

#endif /* NETWORKING_SENDER_H_ */
