/*
 * YaoEvalMaster.cpp
 *
 */

#include "YaoEvalMaster.h"
#include "YaoEvaluator.h"

#include "GC/Machine.hpp"
#include "GC/Program.hpp"
#include "GC/Processor.hpp"
#include "GC/Secret.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "Processor/Instruction.hpp"
#include "YaoWire.hpp"

YaoEvalMaster::YaoEvalMaster(bool continuous, OnlineOptions& opts) :
        ThreadMaster<GC::Secret<YaoEvalWire>>(opts), continuous(continuous)
{
}

GC::Thread<GC::Secret<YaoEvalWire>>* YaoEvalMaster::new_thread(int i)
{
    return new YaoEvaluator(i, *this);
}
