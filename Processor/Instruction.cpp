/*
 * Instruction.cpp
 *
 */

#include "Instruction.h"
#include "instructions.h"
#include "Processor.h"
#include "Math/gf2n.h"
#include "GC/instructions.h"

#include <iomanip>

template<class cgf2n>
void Instruction::execute_clear_gf2n(vector<cgf2n>& registers,
        vector<cgf2n>& memory, ArithmeticProcessor& Proc) const
{
    auto& C2 = registers;
    auto& M2C = memory;
    switch (opcode)
    {
#define X(NAME, PRE, CODE) \
        case NAME: { PRE; for (int i = 0; i < size; i++) { CODE; } } break;
        CLEAR_GF2N_INSTRUCTIONS
#undef X
    }
}

template<class cgf2n>
void Instruction::gbitdec(vector<cgf2n>& registers) const
{
    for (int j = 0; j < size; j++)
    {
        typename cgf2n::internal_type a = registers.at(r[0] + j).get();
        for (unsigned int i = 0; i < start.size(); i++)
        {
            registers.at(start[i] + j) = a & 1;
            a >>= n;
        }
    }
}

template<class cgf2n>
void Instruction::gbitcom(vector<cgf2n>& registers) const
{
    for (int j = 0; j < size; j++)
    {
        typename cgf2n::internal_type a = 0;
        for (unsigned int i = 0; i < start.size(); i++)
        {
            a ^= registers.at(start[i] + j).get() << (i * n);
        }
        registers.at(r[0] + j) = a;
    }
}

void Instruction::execute_regint(ArithmeticProcessor& Proc, vector<Integer>& Mi) const
{
    (void) Mi;
    auto& Ci = Proc.get_Ci();
    switch (opcode)
    {
#define X(NAME, PRE, CODE) \
        case NAME: { PRE; for (int i = 0; i < size; i++) { CODE; } } break;
        REGINT_INSTRUCTIONS
#undef X
    }
}

void Instruction::shuffle(ArithmeticProcessor& Proc) const
{
    for (int i = 0; i < size; i++)
        Proc.write_Ci(r[0] + i, Proc.read_Ci(r[1] + i));
    for (int i = 0; i < size; i++)
    {
        int j = Proc.shared_prng.get_uint(size - i);
        swap(Proc.get_Ci_ref(r[0] + i), Proc.get_Ci_ref(r[0] + i + j));
    }
}

void Instruction::bitdecint(ArithmeticProcessor& Proc) const
{
    for (int j = 0; j < size; j++)
    {
        long a = Proc.read_Ci(r[0] + j);
        for (unsigned int i = 0; i < start.size(); i++)
        {
            Proc.get_Ci_ref(start[i] + j) = (a >> i) & 1;
        }
    }
}

ostream& operator<<(ostream& s, const Instruction& instr)
{
    switch (instr.get_opcode())
    {
#define X(NAME, PRE, CODE) \
    case NAME: s << #NAME; break;
    ALL_INSTRUCTIONS
#undef X
#define X(NAME, CODE) \
    case NAME: s << #NAME; break;
    COMBI_INSTRUCTIONS
    }

    s << " size=" << instr.get_size();
    s << " n=" << instr.get_n();
    s << " r=(";
    for (int i = 0; i < 3; i++)
        s << instr.get_r(i) << ", ";
    s << instr.get_r(3);
    s << ")";
    if (not instr.get_start().empty())
    {
        s << " args=(";
        for (unsigned i = 0; i < instr.get_start().size() - 1; i++)
            s << instr.get_start()[i] << ", ";
        s << instr.get_start().back();
        s << ")";
    }
    return s;
}

template void Instruction::execute_clear_gf2n(vector<gf2n_short>& registers,
        vector<gf2n_short>& memory, ArithmeticProcessor& Proc) const;
template void Instruction::execute_clear_gf2n(vector<gf2n_long>& registers,
        vector<gf2n_long>& memory, ArithmeticProcessor& Proc) const;
