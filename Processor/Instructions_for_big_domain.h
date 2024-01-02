
#ifndef GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H
#define GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H

#define ARITHMETIC_INSTRUCTIONS_FOR_BIG_DOMAIN \
    X(LDI, auto dest = &Procp.get_C()[r[0]]; typename BigDomainShare::clear tmp = int(n), \
            *dest++ = tmp)      \
    X(LDSI, auto dest = &Procp.get_S()[r[0]]; \
            auto tmp = BigDomainShare::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
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
            *dest++ = *op1++ + BigDomainShare::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()))                                            \
    X(ADDM, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ + BigDomainShare::constant(*op2++, Proc.P.my_num(), Procp.MC.get_alphai())) \
    X(ADDCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            typename BigDomainShare::clear op2 = int(n), \
            *dest++ = *op1++ + op2)            \
    X(ADDC, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ + *op2++) \
    X(SUBS, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_S()[r[2]], \
            *dest++ = *op1++ - *op2++)             \
    X(SUBML, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ - BigDomainShare::constant(*op2++, Proc.P.my_num(), Procp.MC.get_alphai())) \
    X(SUBMR, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            auto op2 = &Procp.get_S()[r[2]], \
            *dest++ = BigDomainShare::constant(*op1++, Proc.P.my_num(), Procp.MC.get_alphai()) - *op2++) \
    X(SUBSI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = BigDomainShare::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
            *dest++ = *op1++ - op2)                                                            \
    X(SUBSFI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = BigDomainShare::constant(int(n), Proc.P.my_num(), Procp.MC.get_alphai()), \
            *dest++ = op2 - *op1++)                                                            \
    X(MULM, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            auto op2 = &Procp.get_C()[r[2]], \
            *dest++ = *op1++ * *op2++) \
    X(MULCI, auto dest = &Procp.get_C()[r[0]]; auto op1 = &Procp.get_C()[r[1]]; \
            typename BigDomainShare::clear op2 = int(n), \
            *dest++ = *op1++ * op2) \
    X(MULSI, auto dest = &Procp.get_S()[r[0]]; auto op1 = &Procp.get_S()[r[1]]; \
            typename BigDomainShare::clear op2 = int(n), \
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
            BigDomainShare s, \
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


#ifdef BIG_DOMAIN_FOR_RING

template<class sint, class sgf2n>
inline void Instruction::execute_big_domain_instructions(Processor<sint, sgf2n>& Proc) const
{
  auto& Procp = *Proc.Procp_2;
  switch (opcode) {
    case LDMINT:
      cout << "LDMINT" << endl;
      throw not_implemented();
      break;
    case GLDMC:
      cout << "GLDMC" << endl;
      throw not_implemented();
      break;
    case GLDMS:
      cout << "GLDMC" << endl;
      throw not_implemented();
      break;
    case BITDECINT:
      bitdecint(Proc);
      break;
    case PRINTCHR:
      Proc.out << string((char*)&this->n,1) << flush;
      break;
    case PRINTSTR:
      Proc.out << string((char*)&this->n,4) << flush;
      break;
    case LDMC:
    case REQBL:
    case USE:
    case MULS:
    case OPEN:
    case PRINTREGPLAIN:
    case START:
    case STOP:
    case TRUNC_PR:
    case CSD:
    case STMC:
    case LDMCI:
    case USE_INP:
    case CRASH:
    case GENSECSHUFFLE:
    case APPLYSHUFFLE:
    case DELSHUFFLE:
    case CONDPRINTSTR:
    case STMCI:
      execute(Proc);
      break;
    case PRINTFLOATPLAIN:
      print(Proc.out, &Proc.Procp_2->C[start[0]], &Proc.Procp_2->C[start[1]],
            &Proc.Procp_2->C[start[2]], &Proc.Procp_2->C[start[3]],
            &Proc.Procp_2->C[start[4]]);
      return;
    case CONVMODP:
      if (n == 0)
      {
        for (int i = 0; i < size; i++)
          Proc.write_Ci(r[0] + i,
                        Proc.sync(
                                Integer::convert_unsigned(Proc.Procp_2->C[r[1] + i]).get()));
      }
      else if (n <= 64)
        for (int i = 0; i < size; i++)
          Proc.write_Ci(r[0] + i,
                        Proc.sync(Integer(Proc.Procp_2->C[r[1] + i], n).get()));
      else
        throw Processor_Error(to_string(n) + "-bit conversion impossible; "
                                             "integer registers only have 64 bits");
      return;
    case BIT:
      {

       auto dest = &Procp.get_S()[r[0]];
        for (int i = 0; i < size; i++) {
          Procp.DataF.get_one(DATA_BIT, *dest++);
        }
      }
      break;
#define X(NAME, PRE, CODE) \
        case NAME: { PRE; for (int i = 0; i < size; i++) { CODE; } } break;
      ARITHMETIC_INSTRUCTIONS_FOR_BIG_DOMAIN
#undef X
    default:
      cout << "instruction with code " << opcode << "  is not impletementd by big domain" << endl;
      throw not_implemented();
  }
}

template<class sint, class sgf2n>
inline void Instruction::execute(Processor<sint, sgf2n>& Proc) const
{

  auto& Procp = Proc.Procp;
  auto& Proc2 = Proc.Proc2;

  // optimize some instructions
  switch (opcode)
  {

    case CONVMODP:
      if (n == 0)
        {
          for (int i = 0; i < size; i++)
            Proc.write_Ci(r[0] + i,
                Proc.sync(
                    Integer::convert_unsigned(Proc.read_Cp(r[1] + i)).get()));
        }
      else if (n <= 64)
        for (int i = 0; i < size; i++)
          Proc.write_Ci(r[0] + i,
              Proc.sync(Integer(Proc.read_Cp(r[1] + i), n).get()));
      else
        throw Processor_Error(to_string(n) + "-bit conversion impossible; "
            "integer registers only have 64 bits");
      return;
  }

  int r[3] = {this->r[0], this->r[1], this->r[2]};
  int64_t n = this->n;
  for (int i = 0; i < size; i++)
  { switch (opcode)
    {
      case CSD:
        if (!Proc.change_domain){
//          Proc.Procp.S[r[1]] = Proc.Procp_2->S[r[0]];
          break;
        }else{
//          Proc.Procp_2->S[r[1]] = Proc.Procp.S[r[0]];
          Proc.Procp_2->protocol.change_domain(start, size,  *Proc.Procp_2);
          return;
        }
      case LDMC:
        if (!Proc.change_domain)
          Proc.write_Cp(r[0],Proc.machine.Mp.read_C(n));
        else
          Proc.Procp_2->C[r[0]] = Proc.machine.Mp.read_C(n);
        n++;
        break;
      case LDMCI:
        if (!Proc.change_domain)
          Proc.write_Cp(r[0], Proc.machine.Mp.read_C(Proc.sync_Ci(r[1])));
        else
          Proc.Procp_2->C[r[0]] =  Proc.machine.Mp_2->read_C(Proc.sync_Ci(r[1]));
        break;
      case STMC:
        if (!Proc.change_domain)
          Proc.machine.Mp.write_C(n,Proc.read_Cp(r[0]));
        else
          Proc.machine.Mp_2->write_C(n, Proc.Procp_2->C[r[0]]);

        n++;
        break;
      case STMCI:
        if (!Proc.change_domain)
           Proc.machine.Mp.write_C(Proc.sync_Ci(r[1]), Proc.read_Cp(r[0]));
        else
           Proc.machine.Mp_2->write_C(Proc.sync_Ci(r[1]), Proc.Procp_2->C[r[0]]);

        break;
      case MOVC:
        Proc.write_Cp(r[0],Proc.read_Cp(r[1]));
        break;
      case DIVC:
          Proc.write_Cp(r[0], Proc.read_Cp(r[1]) / sanitize(Proc.Procp, r[2]));

        break;
      case GDIVC:
        Proc.write_C2(r[0], Proc.read_C2(r[1]) / sanitize(Proc.Proc2, r[2]));
        break;
      case FLOORDIVC:
        Proc.temp.aa.from_signed(Proc.read_Cp(r[1]));
          Proc.temp.aa2.from_signed(sanitize(Proc.Procp, r[2]));

        Proc.write_Cp(r[0], bigint(Proc.temp.aa / Proc.temp.aa2));
        break;
      case MODC:
        to_bigint(Proc.temp.aa, Proc.read_Cp(r[1]));

          to_bigint(Proc.temp.aa2, sanitize(Proc.Procp, r[2]));

        mpz_fdiv_r(Proc.temp.aa.get_mpz_t(), Proc.temp.aa.get_mpz_t(), Proc.temp.aa2.get_mpz_t());
        Proc.temp.ansp.convert_destroy(Proc.temp.aa);
        Proc.write_Cp(r[0],Proc.temp.ansp);
        break;
      case LEGENDREC:
        to_bigint(Proc.temp.aa, Proc.read_Cp(r[1]));
        Proc.temp.aa = mpz_legendre(Proc.temp.aa.get_mpz_t(), sint::clear::pr().get_mpz_t());
        to_gfp(Proc.temp.ansp, Proc.temp.aa);
        Proc.write_Cp(r[0], Proc.temp.ansp);
        break;
      case DIGESTC:
      {
        octetStream o;
        to_bigint(Proc.temp.aa, Proc.read_Cp(r[1]));
        to_gfp(Proc.temp.ansp, Proc.temp.aa);
        Proc.temp.ansp.pack(o);
        // keep first n bytes
        to_gfp(Proc.temp.ansp, o.check_sum(n));
        Proc.write_Cp(r[0], Proc.temp.ansp);
      }
        break;
      case DIVCI:
        if (n == 0)
          throw Processor_Error("Division by immediate zero");
        Proc.write_Cp(r[0], Proc.read_Cp(r[1]) / n);
        break;
      case GDIVCI:
        if (n == 0)
          throw Processor_Error("Division by immediate zero");
        Proc.write_C2(r[0], Proc.read_C2(r[1]) / n);
        break;
      case INV2M:
        Proc.write_Cp(r[0], Proc.get_inverse2(n));
        break;
      case MODCI:
        if (n == 0)
          throw Processor_Error("Modulo by immediate zero");
        to_bigint(Proc.temp.aa, Proc.read_Cp(r[1]));
        to_gfp(Proc.temp.ansp, Proc.temp.aa2 = mpz_fdiv_ui(Proc.temp.aa.get_mpz_t(), n));
        Proc.write_Cp(r[0],Proc.temp.ansp);
        break;
      case SQUARE:
          Procp.DataF.get_two(DATA_SQUARE, Proc.get_Sp_ref(r[0]),Proc.get_Sp_ref(r[1]));
        break;
      case GSQUARE:
        Proc2.DataF.get_two(DATA_SQUARE, Proc.get_S2_ref(r[0]),Proc.get_S2_ref(r[1]));
        break;
      case INV:
          Procp.DataF.get_two(DATA_INVERSE, Proc.get_Sp_ref(r[0]),Proc.get_Sp_ref(r[1]));
        break;
      case GINV:
        Proc2.DataF.get_two(DATA_INVERSE, Proc.get_S2_ref(r[0]),Proc.get_S2_ref(r[1]));
        break;
      case RANDOMS:
          Procp.protocol.randoms_inst(Procp.get_S(), *this);
        return;
      case INPUTMASKREG:
        Procp.DataF.get_input(Proc.get_Sp_ref(r[0]), Proc.temp.rrp, Proc.sync_Ci(r[2]));
        Proc.write_Cp(r[1], Proc.temp.rrp);
        break;
      case INPUTMASK:
        Procp.DataF.get_input(Proc.get_Sp_ref(r[0]), Proc.temp.rrp, n);
        Proc.write_Cp(r[1], Proc.temp.rrp);
        break;
      case GINPUTMASK:
        Proc2.DataF.get_input(Proc.get_S2_ref(r[0]), Proc.temp.ans2, n);
        Proc.write_C2(r[1], Proc.temp.ans2);
        break;
      case INPUT:
          sint::Input::template input<IntInput<typename sint::clear>>(Proc.Procp, start, size);
        return;
      case GINPUT:
        sgf2n::Input::template input<IntInput<typename sgf2n::clear>>(Proc.Proc2, start, size);
        return;
      case INPUTFIX:
          sint::Input::template input<FixInput>(Proc.Procp, start, size);
        return;
      case INPUTFLOAT:
          sint::Input::template input<FloatInput>(Proc.Procp, start, size);
        return;
      case INPUTMIXED:
          sint::Input::input_mixed(Proc.Procp, start, size, false);
        return;
      case INPUTMIXEDREG:
          sint::Input::input_mixed(Proc.Procp, start, size, true);
        return;
      case RAWINPUT:
          Proc.Procp.input.raw_input(Proc.Procp, start, size);
        return;
      case GRAWINPUT:
        Proc.Proc2.input.raw_input(Proc.Proc2, start, size);
        return;
      case INPUTPERSONAL:
          Proc.Procp.input_personal(start);
        return;
      case SENDPERSONAL:
          Proc.Procp.send_personal(start);
        return;
      case PRIVATEOUTPUT:
          Proc.Procp.check();
          Proc.Procp.private_output(start);
        return;
      // Note: Fp version has different semantics for NOTC than GNOTC
      case NOTC:
        to_bigint(Proc.temp.aa, Proc.read_Cp(r[1]));
        mpz_com(Proc.temp.aa.get_mpz_t(), Proc.temp.aa.get_mpz_t());
        Proc.temp.aa2 = 1;
        Proc.temp.aa2 <<= n;
        Proc.temp.aa += Proc.temp.aa2;
        Proc.temp.ansp.convert_destroy(Proc.temp.aa);
        Proc.write_Cp(r[0],Proc.temp.ansp);
        break;
      case SHRSI:
          sint::shrsi(Procp, *this);
        return;
      case GSHRSI:
        sgf2n::shrsi(Proc2, *this);
        return;
      case OPEN:
          if (!Proc.change_domain)
            Proc.Procp.POpen(*this);
          else
            Proc.Procp_2->POpen(*this);
        return;
      case GOPEN:
        Proc.Proc2.POpen(*this);
        return;
      case MULS:
        if (!Proc.change_domain)
          Proc.Procp.muls(start, size);
        else
          Proc.Procp_2->muls(start, size);
        return;
      case GMULS:
        Proc.Proc2.protocol.muls(start, Proc.Proc2, Proc.MC2, size);
        return;
      case MULRS:
          Proc.Procp.mulrs(start);
        return;
      case GMULRS:
        Proc.Proc2.protocol.mulrs(start, Proc.Proc2);
        return;
      case DOTPRODS:
          Proc.Procp.dotprods(start, size);
        return;
      case GDOTPRODS:
        Proc.Proc2.dotprods(start, size);
        return;
      case MATMULS:
          Proc.Procp.matmuls(Proc.Procp.get_S(), *this);
        return;
      case MATMULSM:
            Proc.Procp.protocol.matmulsm(Proc.Procp, Proc.machine.Mp.MS, *this);
        return;
      case CONV2DS:
          Proc.Procp.protocol.conv2ds(Proc.Procp, *this);
        return;
      case TRUNC_PR:
        if (!Proc.change_domain)
          Proc.Procp.protocol.trunc_pr(start, size, Proc.Procp);
        else
          Proc.Procp_2->protocol.trunc_pr(start, size, *Proc.Procp_2);
        return;

      case SECSHUFFLE:
          Proc.Procp.secure_shuffle(*this);
        return;
      case GSECSHUFFLE:
        Proc.Proc2.secure_shuffle(*this);
        return;
      case GENSECSHUFFLE:
        if (!Proc.change_domain)
          Proc.write_Ci(r[0], Proc.Procp.generate_secure_shuffle(*this));
        else
          Proc.write_Ci(r[0], Proc.Procp_2->generate_secure_shuffle(*this));
        return;
      case APPLYSHUFFLE:
        if (!Proc.change_domain)
          Proc.Procp.apply_shuffle(*this, Proc.read_Ci(start.at(3)));
        else
          Proc.Procp_2->apply_shuffle(*this, Proc.read_Ci(start.at(3)));
        return;
      case DELSHUFFLE:
        if (!Proc.change_domain)
          Proc.Procp.delete_shuffle(Proc.read_Ci(r[0]));
        else
          Proc.Procp_2->delete_shuffle(Proc.read_Ci(r[0]));
        return;
      case INVPERM:

          Proc.Procp.inverse_permutation(*this);

        return;
      case CHECK:
        {
          CheckJob job;
          if (BaseMachine::thread_num == 0)
            BaseMachine::s().queues.distribute(job, 0);
          Proc.check();
          if (BaseMachine::thread_num == 0)
            BaseMachine::s().queues.wrap_up(job);
          return;
        }
      case JMP:
        Proc.PC += (signed int) n;
        break;
      case JMPI:
        Proc.PC += (signed int) Proc.sync_Ci(r[0]);
        break;
      case JMPNZ:
        if (Proc.sync_Ci(r[0]) != 0)
          { Proc.PC += (signed int) n; }
        break;
      case JMPEQZ:
        if (Proc.sync_Ci(r[0]) == 0)
          { Proc.PC += (signed int) n; }
        break;
      case PRINTREG:
           {
             Proc.out << "Reg[" << r[0] << "] = " << Proc.read_Cp(r[0])
              << " # " << string((char*)&n, 4) << endl;
           }
        break;
      case PRINTREGPLAIN:
        if (!Proc.change_domain)
          print(Proc.out, &Proc.read_Cp(r[0]));
        else
          print(Proc.out, &Proc.Procp_2->C[r[0]]);
        return;
      case CONDPRINTPLAIN:
        if (not Proc.read_Cp(r[0]).is_zero())
          {
            print(Proc.out, &Proc.read_Cp(r[1]), &Proc.read_Cp(r[2]));
          }
        return;
      case PRINTFLOATPLAIN:
        print(Proc.out, &Proc.read_Cp(start[0]), &Proc.read_Cp(start[1]),
            &Proc.read_Cp(start[2]), &Proc.read_Cp(start[3]),
            &Proc.read_Cp(start[4]));
        return;
      case CONDPRINTSTR:
        if (!Proc.change_domain)
          {
            if (not Proc.read_Cp(r[0]).is_zero())
              {
                string str = {(char*)&n, 4};
                size_t n = str.find('\0');
                if (n < 4)
                  str.erase(n);
                Proc.out << str << flush;
              }
          }
        else {
          if (not Proc.Procp_2->C[r[0]].is_zero())
              {
                string str = {(char*)&n, 4};
                size_t n = str.find('\0');
                if (n < 4)
                  str.erase(n);
                Proc.out << str << flush;
              }
        }
        break;
      case REQBL:
      case GREQBL:
      case USE:
      case USE_INP:
      case USE_EDABIT:
      case USE_MATMUL:
      case USE_PREP:
      case GUSE_PREP:
        break;
      case TIME:
        Proc.machine.time();
	break;
      case START:
        Proc.machine.set_thread_comm(Proc.P.total_comm());
        Proc.machine.start(n);
        break;
      case STOP:
        Proc.machine.set_thread_comm(Proc.P.total_comm());
        Proc.machine.stop(n);
        break;
      case RUN_TAPE:
        Proc.machine.run_tapes(start, Proc.DataF);
        break;
      case JOIN_TAPE:
        Proc.machine.join_tape(r[0]);
        break;
      case CRASH:
        if (Proc.sync_Ci(r[0]))
          throw crash_requested();
        break;
      case STARTGRIND:
        CALLGRIND_START_INSTRUMENTATION;
        break;
      case STOPGRIND:
        CALLGRIND_STOP_INSTRUMENTATION;
        break;
      case NPLAYERS:
        Proc.write_Ci(r[0], Proc.P.num_players());
        break;
      case THRESHOLD:
        Proc.write_Ci(r[0], sint::threshold(Proc.P.num_players()));
        break;
      case PLAYERID:
        Proc.write_Ci(r[0], Proc.P.my_num());
        break;
      // ***
      // TODO: read/write shared GF(2^n) data instructions
      // ***
      case LISTEN:
        // listen for connections at port number n
        Proc.external_clients.start_listening(Proc.sync_Ci(r[0]));
        break;
      case ACCEPTCLIENTCONNECTION:
      {
        // get client connection at port number n + my_num())
        int client_handle = Proc.external_clients.get_client_connection(
            Proc.read_Ci(r[1]));
        if (Proc.P.my_num() == 0)
        {
          octetStream os;
          os.store(int(sint::open_type::type_char()));
          sint::specification(os);
          os.Send(Proc.external_clients.get_socket(client_handle));
        }
        Proc.write_Ci(r[0], client_handle);
        break;
      }
      case CLOSECLIENTCONNECTION:
        Proc.external_clients.close_connection(Proc.read_Ci(r[0]));
        break;
      case READSOCKETINT:
        Proc.read_socket_ints(Proc.read_Ci(r[0]), start, n);
        break;
      case READSOCKETC:
        Proc.read_socket_vector(Proc.read_Ci(r[0]), start, n);
        break;
      case READSOCKETS:
        // read shares and MAC shares
        Proc.read_socket_private(Proc.read_Ci(r[0]), start, n, true);
        break;
      case WRITESOCKETINT:
        Proc.write_socket(INT, false, Proc.read_Ci(r[0]), r[1], start, n);
        break;
      case WRITESOCKETC:
        Proc.write_socket(CINT, false, Proc.read_Ci(r[0]), r[1], start, n);
        break;
      case WRITESOCKETS:
        // Send shares + MACs
        Proc.write_socket(SINT, true, Proc.read_Ci(r[0]), r[1], start, n);
        break;
      case WRITESOCKETSHARE:
        // Send only shares, no MACs
        // N.B. doesn't make sense to have a corresponding read instruction for this
        Proc.write_socket(SINT, false, Proc.read_Ci(r[0]), r[1], start, n);
        break;
      case WRITEFILESHARE:
        // Write shares to file system
        Proc.write_shares_to_file(Proc.read_Ci(r[0]), start);
        break;
      case READFILESHARE:
        // Read shares from file system
        Proc.read_shares_from_file(Proc.read_Ci(r[0]), r[1], start);
        break;
      case PUBINPUT:
        Proc.get_Cp_ref(r[0]) = Proc.template
            get_input<IntInput<typename sint::clear>>(
            Proc.public_input, Proc.public_input_filename, 0).items[0];
        break;
      case RAWOUTPUT:
        Proc.read_Cp(r[0]).output(Proc.public_output, false);
        break;
      case INTOUTPUT:
        if (n == -1 or n == Proc.P.my_num())
          Integer(Proc.read_Ci(r[0])).output(Proc.binary_output, false);
        break;
      case FLOATOUTPUT:
        if (n == -1 or n == Proc.P.my_num())
          {
            double tmp = bigint::get_float(Proc.read_Cp(start[0] + i),
              Proc.read_Cp(start[1] + i), Proc.read_Cp(start[2] + i),
              Proc.read_Cp(start[3] + i)).get_d();
            Proc.binary_output.write((char*) &tmp, sizeof(double));
          }
        break;
      case PREP:

          Procp.DataF.get(Proc.Procp.get_S(), r, start, size);

        return;
      case GPREP:
        Proc2.DataF.get(Proc.Proc2.get_S(), r, start, size);
        return;
      case CISC:

          Procp.protocol.cisc(Procp, *this);

        return;
      default:
        printf("Case of opcode=0x%x not implemented yet\n",opcode);
        throw invalid_opcode(opcode);
        break;
#define X(NAME, CODE) case NAME:
        COMBI_INSTRUCTIONS
#undef X
#define X(NAME, PRE, CODE) case NAME:
        ARITHMETIC_INSTRUCTIONS
#undef X
#define X(NAME, PRE, CODE) case NAME:
        CLEAR_GF2N_INSTRUCTIONS
#undef X
#define X(NAME, PRE, CODE) case NAME:
        REGINT_INSTRUCTIONS
#undef X
        throw runtime_error("wrong case statement"); return;
    }
  if (size > 1)
    {
      r[0]++; r[1]++; r[2]++;
    }
  }
}

template<class sint, class sgf2n>
void Program::execute(Processor<sint, sgf2n>& Proc) const
{
  unsigned int size = p.size();
  Proc.PC=0;

  auto& Procp = Proc.Procp;
  auto& Proc2 = Proc.Proc2;

  // binary instructions
  typedef typename sint::bit_type T;
  auto& processor = Proc.Procb;
  auto& Ci = Proc.get_Ci();

  while (Proc.PC<size)
    {
      auto& instruction = p[Proc.PC];
      auto& r = instruction.r;
      auto& n = instruction.n;
      auto& start = instruction.start;
      auto& size = instruction.size;
      (void) start;

#ifdef COUNT_INSTRUCTIONS
#ifdef TIME_INSTRUCTIONS
      RunningTimer timer;
      int PC = Proc.PC;
#else
      Proc.stats[p[Proc.PC].get_opcode()]++;
#endif
#endif

#ifdef OUTPUT_INSTRUCTIONS
      cerr << instruction << endl;
#endif

      Proc.PC++;
      if (instruction.get_opcode() == CMD){
        // only work when T is Rep3Share and one of the small domain size is smaller than 2^32
        if (!Proc.change_domain){
          Proc.change_domain = true;
          Proc.start_subprocessor_for_big_domain();
          auto& Procp_for_big_domain = *(Proc.Procp_2);
          Procp_for_big_domain.template assign_S<sint>(Procp.get_S());
          Procp_for_big_domain.template assign_C<sint>(Procp.get_C());
//          cout << "Proc.machine.Mp.MS.size() = " << Proc.machine.Mp.MS.size() << endl;
//          cout << "Proc.machine.Mp_2->MS.size() = " << Proc.machine.Mp_2->MS.size() << endl;
        }
        else{
          Proc.change_domain = false;
          auto& Procp_for_big_domain = *(Proc.Procp_2);
          Procp.template assign_S<BigDomainShare>(Procp_for_big_domain.get_S());
          Procp.template assign_C<BigDomainShare>(Procp_for_big_domain.get_C());
          Proc.stop_subprocessor_for_big_domain();
        }
        continue;
      }
      if (Proc.change_domain){
        instruction.execute_big_domain_instructions(Proc);
        continue;
      }
      switch(instruction.get_opcode())
        {
#define X(NAME, PRE, CODE) \
        case NAME: { PRE; for (int i = 0; i < size; i++) { CODE; } } break;
        ARITHMETIC_INSTRUCTIONS
#undef X
#define X(NAME, PRE, CODE) case NAME:
        CLEAR_GF2N_INSTRUCTIONS
        instruction.execute_clear_gf2n(Proc2.get_C(), Proc.machine.M2.MC, Proc); break;
#undef X
#define X(NAME, PRE, CODE) case NAME:
        REGINT_INSTRUCTIONS
        instruction.execute_regint(Proc, Proc.machine.Mi.MC); break;
#undef X
#define X(NAME, CODE) case NAME: CODE; break;
        COMBI_INSTRUCTIONS
#undef X
        default:
          instruction.execute(Proc);
        }

#if defined(COUNT_INSTRUCTIONS) and defined(TIME_INSTRUCTIONS)
      Proc.stats[p[PC].get_opcode()] += timer.elapsed() * 1e9;
#endif
    }
}

#endif

#endif //GARNET_INSTRUCTIONS_FOR_BIG_DOMAIN_H
