/*
 * simple-offline.cpp
 *
 */

#include <FHEOffline/SimpleMachine.h>
#include "Tools/callgrind.h"

int main(int argc, const char** argv)
{
    CALLGRIND_STOP_INSTRUMENTATION;
    SimpleMachine machine(argc, argv);
    CALLGRIND_START_INSTRUMENTATION;
    machine.run();
}
