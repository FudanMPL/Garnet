/*
 * PostSacriRepFieldShare.h
 *
 */

#ifndef PROTOCOLS_POSTSACRIREPFIELDSHARE_H_
#define PROTOCOLS_POSTSACRIREPFIELDSHARE_H_

#include "Protocols/MaliciousRep3Share.h"

template<class T> class MalRepRingPrepWithBits;
template<class T> class PostSacrifice;

template<class T>
class PostSacriRepFieldShare : public MaliciousRep3Share<T>
{
    typedef MaliciousRep3Share<T> super;
    typedef PostSacriRepFieldShare This;

public:
    typedef typename super::clear clear;

    typedef PostSacrifice<PostSacriRepFieldShare> Protocol;
    typedef HashMaliciousRepMC<PostSacriRepFieldShare> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef ReplicatedInput<PostSacriRepFieldShare> Input;
    typedef ReplicatedPO<This> PO;
    typedef SpecificPrivateOutput<This> PrivateOutput;
    typedef MaliciousRepPrepWithBits<PostSacriRepFieldShare> LivePrep;

    PostSacriRepFieldShare()
    {
    }
    PostSacriRepFieldShare(const clear& other, int my_num, clear alphai = {}) :
            super(other, my_num, alphai)
    {
    }
    template<class U>
    PostSacriRepFieldShare(const U& other) : super(other)
    {
    }
};

#endif /* PROTOCOLS_POSTSACRIREPFIELDSHARE_H_ */
