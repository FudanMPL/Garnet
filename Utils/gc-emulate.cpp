/*
 * gc-emulate.cpp
 *
 */

#include <GC/Program.h>
#include <GC/Instruction.h>
#include <GC/FakeSecret.h>
#include "GC/Machine.h"
#include "GC/Processor.h"

#include "GC/Processor.hpp"
#include "GC/Machine.hpp"
#include "GC/Program.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "Processor/Machine.hpp"
#include "Processor/Instruction.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
        exit(1);

    GC::Memory<GC::FakeSecret> dynamic_memory;
    GC::Machine<GC::FakeSecret> machine;
    GC::Processor<GC::FakeSecret> processor(machine);
    GC::Program program;
    program.parse(string(argv[1]) + "-0");
    machine.reset(program, dynamic_memory);
    processor.reset(program);
    if (argc > 2)
        processor.open_input_file(argv[2]);
    while (program.execute(processor, dynamic_memory) != GC::DONE_BREAK);
}
