/*
 * YaoEvalMaster.h
 *
 */

#ifndef YAO_YAOEVALMASTER_H_
#define YAO_YAOEVALMASTER_H_

#include "GC/ThreadMaster.h"
#include "GC/Secret.h"
#include "YaoEvalWire.h"

class YaoEvalMaster : public GC::ThreadMaster<GC::Secret<YaoEvalWire>>
{
public:
    bool continuous;

    YaoEvalMaster(bool continuous, OnlineOptions& opts);

    GC::Thread<GC::Secret<YaoEvalWire>>* new_thread(int i);
};

#endif /* YAO_YAOEVALMASTER_H_ */
