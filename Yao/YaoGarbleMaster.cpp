/*
 * YaoGarbleMaster.cpp
 *
 */

#include "YaoGarbleMaster.h"
#include "YaoGarbler.h"

#include "GC/Machine.hpp"
#include "GC/Program.hpp"
#include "GC/Processor.hpp"
#include "GC/Secret.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "Processor/Instruction.hpp"
#include "YaoWire.hpp"

YaoGarbleMaster::YaoGarbleMaster(bool continuous, OnlineOptions& opts, int threshold) :
        super(opts), continuous(continuous), threshold(threshold)
{
    PRNG G;
    G.ReSeed();
    delta = G.get_doubleword();
    delta.set_signal(1);
}

GC::Thread<GC::Secret<YaoGarbleWire>>* YaoGarbleMaster::new_thread(int i)
{
    return new YaoGarbler(i, *this);
}
