/*
 * instructions.h
 *
 */

#ifndef GC_INSTRUCTIONS_H_
#define GC_INSTRUCTIONS_H_

#include "Tools/callgrind.h"
#include "Instruction.h"

#define PROC processor
#define INST instruction
#define MACH processor.machine
#define MD dynamic_memory

#define R0 instruction.get_r(0)
#define REG1 instruction.get_r(1)

#define S0 processor.S[instruction.get_r(0)]
#define PS1 processor.S[instruction.get_r(1)]
#define PS2 processor.S[instruction.get_r(2)]

#define C0 processor.C[instruction.get_r(0)]
#define PC1 processor.C[instruction.get_r(1)]
#define PC2 processor.C[instruction.get_r(2)]

#define I0 processor.I[instruction.get_r(0)]
#define PI1 processor.I[instruction.get_r(1)]
#define PI2 processor.I[instruction.get_r(2)]

#define IMM instruction.get_n()
#define EXTRA instruction.get_start()
#define SIZE instruction.get_size()

#define MMS processor.memories.MS
#define MMC processor.memories.MC
#define MID MACH->MI[IMM]
#define MII MACH->MI[PI1.get()]

#define BIT_INSTRUCTIONS \
    X(XORS, T::xors(PROC, EXTRA)) \
    X(XORCB, processor.xorc(instruction)) \
    X(XORCBI, C0.xor_(PC1, IMM)) \
    X(NOTS, processor.nots(INST)) \
    X(NOTCB, processor.notcb(INST)) \
    X(ANDRS, T::andrs(PROC, EXTRA)) \
    X(ANDRSVEC, T::andrsvec(PROC, EXTRA)) \
    X(ANDS, T::ands(PROC, EXTRA)) \
    X(ANDM, T::andm(PROC, instruction)) \
    X(ADDCB, C0 = PC1 + PC2) \
    X(ADDCBI, C0 = PC1 + int(IMM)) \
    X(MULCBI, C0 = PC1 * int(IMM)) \
    X(BITDECS, PROC.bitdecs(EXTRA, S0)) \
    X(BITCOMS, PROC.bitcoms(S0, EXTRA)) \
    X(BITDECC, PROC.bitdecc(EXTRA, C0)) \
    X(SHRCBI, C0 = PC1 >> IMM) \
    X(SHLCBI, C0 = PC1 << IMM) \
    X(LDBITS, S0.load_clear(REG1, int(IMM))) \
    X(LDMSB, PROC.mem_op(SIZE, PROC.S, MMS, R0, IMM)) \
    X(STMSB, PROC.mem_op(SIZE, MMS, PROC.S, IMM, R0)) \
    X(LDMCB, PROC.mem_op(SIZE, PROC.C, MMC, R0, IMM)) \
    X(STMCB, PROC.mem_op(SIZE, MMC, PROC.C, IMM, R0)) \
    X(LDMSBI, PROC.mem_op(SIZE, PROC.S, MMS, R0, Ci[REG1])) \
    X(STMSBI, PROC.mem_op(SIZE, MMS, PROC.S, Ci[REG1], R0)) \
    X(LDMCBI, PROC.mem_op(SIZE, PROC.C, MMC, R0, Ci[REG1])) \
    X(STMCBI, PROC.mem_op(SIZE, MMC, PROC.C, Ci[REG1], R0)) \
    X(MOVSB, S0 = PS1) \
    X(TRANS, T::trans(PROC, IMM, EXTRA)) \
    X(BITB, PROC.random_bit(S0)) \
    X(REVEAL, T::reveal_inst(PROC, EXTRA)) \
    X(PRINTREGSIGNED, PROC.print_reg_signed(IMM, R0)) \
    X(PRINTREGB, PROC.print_reg(R0, IMM, SIZE)) \
    X(PRINTREGPLAINB, PROC.print_reg_plain(C0)) \
    X(PRINTFLOATPLAINB, PROC.print_float(EXTRA)) \
    X(CONDPRINTSTRB, if(C0.get()) PROC.print_str(IMM)) \

#define COMBI_INSTRUCTIONS BIT_INSTRUCTIONS \
    X(INPUTB, T::inputb(PROC, Proc, EXTRA)) \
    X(INPUTBVEC, T::inputbvec(PROC, Proc, EXTRA)) \
    X(CONVSINT, S0.load_clear(IMM, Proc.read_Ci(REG1))) \
    X(CONVCINT, C0 = Proc.read_Ci(REG1)) \
    X(CONVCBIT, Proc.write_Ci(R0, PC1.get())) \
    X(CONVCINTVEC, Proc.convcintvec(instruction)) \
    X(CONVCBITVEC, Proc.convcbitvec(instruction)) \
    X(CONVCBIT2S, PROC.convcbit2s(instruction)) \
    X(DABIT, Proc.dabit(INST)) \
    X(EDABIT, Proc.edabit(INST)) \
    X(SEDABIT, Proc.edabit(INST, true)) \
    X(SPLIT, Proc.split(INST)) \

#define GC_INSTRUCTIONS \
    X(INPUTB, T::inputb(PROC, EXTRA)) \
    X(INPUTBVEC, T::inputbvec(PROC, PROC, EXTRA)) \
    X(LDMSD, PROC.load_dynamic_direct(EXTRA, MD)) \
    X(STMSD, PROC.store_dynamic_direct(EXTRA, MD)) \
    X(LDMSDI, PROC.load_dynamic_indirect(EXTRA, MD)) \
    X(STMSDI, PROC.store_dynamic_indirect(EXTRA, MD)) \
    X(STMSDCI, PROC.store_clear_in_dynamic(EXTRA, MD)) \
    X(CONVSINT, S0.load_clear(IMM, PI1)) \
    X(CONVCINT, C0 = PI1) \
    X(CONVCBIT, T::convcbit(I0, PC1, PROC)) \
    X(CONVCBIT2S, T::convcbit2s(PROC, instruction)) \
    X(PRINTCHR, PROC.print_chr(IMM)) \
    X(PRINTSTR, PROC.print_str(IMM)) \
    X(PRINTFLOATPREC, PROC.print_float_prec(IMM)) \
    X(LDINT, I0 = int(IMM)) \
    X(ADDINT, I0 = PI1 + PI2) \
    X(SUBINT, I0 = PI1 - PI2) \
    X(MULINT, I0 = PI1 * PI2) \
    X(DIVINT, I0 = PI1 / PI2) \
    X(JMP, PROC.PC += IMM) \
    X(JMPNZ, if (I0 != 0) PROC.PC += IMM) \
    X(JMPEQZ, if (I0 == 0) PROC.PC += IMM) \
    X(EQZC, I0 = PI1 == 0) \
    X(LTZC, I0 = PI1 < 0) \
    X(LTC, I0 = PI1 < PI2) \
    X(GTC, I0 = PI1 > PI2) \
    X(EQC, I0 = PI1 == PI2) \
    X(JMPI, PROC.PC += I0) \
    X(LDMINT, I0 = MID) \
    X(STMINT, MID = I0) \
    X(LDMINTI, I0 = MII) \
    X(STMINTI, MII = I0) \
    X(PUSHINT, PROC.pushi(I0.get())) \
    X(POPINT, long x; PROC.popi(x); I0 = x) \
    X(MOVINT, I0 = PI1) \
    X(BITDECINT, PROC.bitdecint(EXTRA, I0)) \
    X(LDARG, I0 = PROC.get_arg()) \
    X(STARG, PROC.set_arg(I0.get())) \
    X(TIME, MACH->time()) \
    X(START, MACH->start(IMM)) \
    X(STOP, MACH->stop(IMM)) \
    X(GLDMS, ) \
    X(GLDMC, ) \
    X(LDMS, ) \
    X(LDMC, ) \
    X(PRINTINT, PROC.out << I0) \
    X(STARTGRIND, CALLGRIND_START_INSTRUMENTATION) \
    X(STOPGRIND, CALLGRIND_STOP_INSTRUMENTATION) \
    X(RUN_TAPE, MACH->run_tapes(EXTRA)) \
    X(JOIN_TAPE, MACH->join_tape(R0)) \
    X(USE, ) \
    X(USE_INP, ) \
    X(NPLAYERS, I0 = Thread<T>::s().P->num_players()) \
    X(THRESHOLD, I0 = T::threshold(Thread<T>::s().P->num_players())) \
    X(PLAYERID, I0 = Thread<T>::s().P->my_num()) \
    X(CRASH, if (I0.get()) throw crash_requested()) \

#define INSTRUCTIONS BIT_INSTRUCTIONS GC_INSTRUCTIONS

#endif /* GC_INSTRUCTIONS_H_ */
