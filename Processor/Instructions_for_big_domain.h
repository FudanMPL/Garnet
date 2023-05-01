//
// Created by 林国鹏 on 2023/3/2.
//

#ifndef GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H
#define GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H

#define ARITHMETIC_INSTRUCTIONS_FOR_BIG_DOMAIN \
    X(LDI, auto dest = &Procp.get_C()[r[0]]; typename Rep3Share128::clear tmp = int(n), \
            *dest++ = tmp)      \
    X(LDSI, auto dest = &Procp.get_S()[r[0]]; \
            auto tmp = Rep3Share128::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
            *dest++ = tmp)                     \
    X(LDMS, auto dest = &Procp.get_S()[r[0]]; auto source = &Proc.machine.Mp_2->MS[n], \
            *dest++ = *source++) \
    X(STMS, auto source = &Procp.get_S()[r[0]]; auto dest = &Proc.machine.Mp_2->MS[n], \
            *dest++ = *source++) \
    X(MOVS, auto dest = &Procp.get_S()[r[0]]; auto source = &Procp.get_S()[r[1]], \
            *dest++ = *source++)\
    X(ADDS, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_S()[r[2]], \
            *dest++ = *op1++ + *op2++)         \
    X(ADDSI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]], \
            *dest++ = *op1++ + Rep3Share128::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()))                                            \
    X(ADDM, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ + Rep3Share128::constant(*op2++, Proc.P.my_num(), Procp.MC.get_alphai())) \
    X(ADDCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            typename Rep3Share128::clear op2 = int(n), \
            *dest++ = *op1++ + op2)            \
    X(ADDC, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ + *op2++) \
    X(SUBS, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_S()[r[2]], \
            *dest++ = *op1++ - *op2++)             \
    X(SUBML, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ - Rep3Share128::constant(*op2++, Proc.P.my_num(), Procp.MC.get_alphai())) \
    X(SUBMR, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            auto op2 = &Procp.get_S()[r[2]], \
            *dest++ = Rep3Share128::constant(*op1++, Proc.P.my_num(), Procp.MC.get_alphai()) - *op2++) \
    X(SUBSI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = Rep3Share128::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
            *dest++ = *op1++ - op2)                                                            \
    X(SUBSFI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = Rep3Share128::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
            *dest++ = op2 - *op1++)                                                            \
    X(MULM, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ * *op2++) \
    X(MULCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            typename Rep3Share128::clear op2 = int(n), \
            *dest++ = *op1++ * op2) \
    X(MULSI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            typename Rep3Share128::clear op2 = int(n), \
            *dest++ = *op1++ * op2)            \
    X(SHLCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]], \
            *dest++ = *op1++ << n) \
    X(SHRCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]], \
            *dest++ = *op1++ >> n)             \
    X(CONVINT, auto dest = &Procp.get_C()[r[0]]; auto source = &Proc.get_Ci()[r[1]], \
            *dest++ = *source++) \
    X(MULC, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ * *op2++) \
    X(LDMSI, auto dest = &Procp.get_S()[r[0]]; auto source = &Proc.get_Ci()[r[1]], \
            *dest++ = Proc.machine.Mp_2->read_S(*source++))                             \
    X(STMSI, auto source = &Procp.get_S()[r[0]]; auto dest = &Proc.get_Ci()[r[1]], \
            Proc.machine.Mp_2->write_S(*dest++, *source++))                             \
    X(PREFIXSUMS, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            Rep3Share128 s, \
            s += *op1++; *dest++ = s)          \
    X(LDINT, auto dest = &Proc.get_Ci()[r[0]], \
            *dest++ = int(n))                  \
    X(ADDINT, auto dest = &Proc.get_Ci()[r[0]]; auto op1 = &Proc.get_Ci()[r[1]]; \
            auto op2 = &Proc.get_Ci()[r[2]], \
            *dest++ = *op1++ + *op2++)         \
    X(INCINT, auto dest = &Proc.get_Ci()[r[0]]; auto base = Proc.get_Ci()[r[1]], \
            int inc = (i / start[0]) % start[1]; *dest++ = base + inc * int(n))         \
    X(LDTN, auto dest = &Proc.get_Ci()[r[0]], *dest++ = Proc.get_thread_num())          \
    X(GTC, auto dest = &Proc.get_Ci()[r[0]]; auto op1 = &Proc.get_Ci()[r[1]]; auto op2 = &Proc.get_Ci()[r[2]], \
            *dest++ = *op1++ > *op2++)         \
    X(SUBINT, auto dest = &Proc.get_Ci()[r[0]]; auto op1 = &Proc.get_Ci()[r[1]]; \
            auto op2 = &Proc.get_Ci()[r[2]], \
            *dest++ = *op1++ - *op2++)         \
    X(MULINT, auto dest = &Proc.get_Ci()[r[0]]; auto op1 = &Proc.get_Ci()[r[1]]; \
            auto op2 = &Proc.get_Ci()[r[2]], \
            *dest++ = *op1++ * *op2++)         \
    X(MOVINT, auto dest = &Proc.get_Ci()[r[0]]; auto source = &Proc.get_Ci()[r[1]], \
            *dest++ = *source++)               \
                                               \
//    X(BIT, auto dest = &Procp.get_S()[r[0]],
//            Procp.DataF.get_one(DATA_BIT, *dest++))




#endif //GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H
