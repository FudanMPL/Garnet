/*
 * PersonalPrep.h
 *
 */

#ifndef GC_PERSONALPREP_H_
#define GC_PERSONALPREP_H_

#include "Protocols/ReplicatedPrep.h"
#include "ShareThread.h"

namespace GC
{

template<class T>
class PersonalPrep : public BufferPrep<T>
{
protected:
    static const int SECURE = -1;

    const int input_player;

    void buffer_personal_triples();

public:
    PersonalPrep(DataPositions& usage, int input_player);

    void buffer_personal_triples(size_t n, ThreadQueues* queues = 0);
    void buffer_personal_triples(vector<array<T, 3>>& triples, size_t begin,
            size_t end);
};

}

#endif /* GC_PERSONALPREP_H_ */
