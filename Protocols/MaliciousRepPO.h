/*
 * MaliciousRepPO.h
 *
 */

#ifndef PROTOCOLS_MALICIOUSREPPO_H_
#define PROTOCOLS_MALICIOUSREPPO_H_

#include "Networking/Player.h"

template<class T>
class MaliciousRepPO
{
protected:
    Player& P;
    octetStream to_send;
    octetStream to_receive[2];
    PointerVector<T> secrets;

public:
    MaliciousRepPO(Player& P);
    virtual ~MaliciousRepPO() {}

    void prepare_sending(const T& secret, int player);
    virtual void send(int player);
    virtual void receive();
    typename T::clear finalize(const T& secret);
    typename T::clear finalize();
};

#endif /* PROTOCOLS_MALICIOUSREPPO_H_ */
