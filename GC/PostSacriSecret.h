/*
 * AbfllnowShare.h
 *
 */

#ifndef GC_POSTSACRISECRET_H_
#define GC_POSTSACRISECRET_H_

#include "MaliciousRepSecret.h"

namespace GC
{

class PostSacriBin;

class PostSacriSecret : public MalRepSecretBase<PostSacriSecret>
{
    typedef PostSacriSecret This;
    typedef MalRepSecretBase<This> super;

public:
    typedef PostSacriBin Protocol;

    PostSacriSecret()
    {
    }

    template<class T>
    PostSacriSecret(const T& other) :
            super(other)
    {
    }
};

}

#endif
