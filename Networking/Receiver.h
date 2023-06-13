/*
 * Receiver.h
 *
 */

#ifndef NETWORKING_RECEIVER_H_
#define NETWORKING_RECEIVER_H_

#include <pthread.h>

#include "Tools/octetStream.h"
#include "Tools/WaitQueue.h"
#include "Tools/time-func.h"

template<class T>
class Receiver
{
    T socket;
    WaitQueue<octetStream*> in;
    WaitQueue<octetStream*> out;
    pthread_t thread;

    static void* run_thread(void* receiver);

    // prevent copying
    Receiver(const Receiver& other);

    void start();
    void stop();
    void run();

public:
    Timer timer;

    Receiver(T socket);
    ~Receiver();

    T get_socket()
    {
        return socket;
    }

    void request(octetStream& os);
    void wait(octetStream& os);
};

#endif /* NETWORKING_RECEIVER_H_ */
