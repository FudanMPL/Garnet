/*
 * Instruction.cpp
 *
 */

#ifndef GC_INSTRUCTION_HPP_
#define GC_INSTRUCTION_HPP_

#include <algorithm>

#include "GC/Instruction.h"
#include "GC/Processor.h"

#include "Processor/Instruction.h"

#include "Tools/parse.h"

#include "GC/Instruction_inline.h"

#include "Processor/Instruction.hpp"

namespace GC
{

Instruction::Instruction() :
        BaseInstruction()
{
    size = 1;
}

bool Instruction::get_offline_data_usage(int& usage)
{
    switch (opcode)
    {
    case ::USE:
        usage += n;
        return int(n) >= 0;
    default:
        return true;
    }
}

unsigned Instruction::get_mem(RegType reg_type) const
{
    unsigned m = n + 1;
    switch (opcode)
    {
    case LDMSD:
        if (reg_type == DYN_SBIT)
        {
            m = 0;
            for (size_t i = 0; i < start.size() / 3; i++)
                m = max(m, (unsigned)start[3*i+1] + 1);
            return m;
        }
        break;
    case STMSD:
        if (reg_type == DYN_SBIT)
        {
            m = 0;
            for (size_t i = 0; i < start.size() / 2; i++)
                m = max(m, (unsigned)start[2*i+1] + 1);
            return m;
        }
        break;
    default:
        return BaseInstruction::get_mem(reg_type);
    }

    return 0;
}

void Instruction::parse(istream& s, int pos)
{
    BaseInstruction::parse(s, pos);

    switch(opcode)
    {
#define X(NAME, CODE) case NAME: \
      break;
    INSTRUCTIONS
#undef X
    default:
        ostringstream os;
        os << "Code not defined for instruction " << showbase << hex << opcode << dec << endl;
        os << "This virtual machine executes binary circuits only." << endl;
        os << "Use 'compile.py -B'." << endl;
        throw Invalid_Instruction(os.str());
        break;
    }
}

} /* namespace GC */

#endif
