
#include "Processor/Program.h"
#include "Processor/Data_Files.h"
#include "Processor/Processor.h"

#include "Processor/Instruction.hpp"

void Program::compute_constants()
{
  for (int reg_type = 0; reg_type < MAX_REG_TYPE; reg_type++)
    {
      max_reg[reg_type] = 0;
      max_mem[reg_type] = 0;
    }
  for (unsigned int i=0; i<p.size(); i++)
    {
      if (!p[i].get_offline_data_usage(offline_data_used))
        unknown_usage = true;
      for (int reg_type = 0; reg_type < MAX_REG_TYPE; reg_type++)
        {
          max_reg[reg_type] = max(max_reg[reg_type],
              p[i].get_max_reg(reg_type));
          max_mem[reg_type] = max(max_mem[reg_type],
              p[i].get_mem(RegType(reg_type)));
        }
      writes_persistence |= p[i].opcode == WRITEFILESHARE;
    }
}

void Program::parse(string filename)
{
  ifstream pinp(filename);
  if (pinp.fail())
    throw file_error(filename);
  parse(pinp);
}

void Program::parse(istream& s)
{
  p.resize(0);
  Instruction instr;
  s.peek();
  while (!s.eof())
    { instr.parse(s, p.size());
      if (s.fail())
        throw runtime_error("error while parsing " + to_string(instr.opcode));
      p.push_back(instr);
      //cerr << "\t" << instr << endl;
      s.peek();
    }
  compute_constants();
}

void Program::print_offline_cost() const
{
  if (unknown_usage)
    {
      cerr << "Tape has unknown usage" << endl;
      return;
    }

  cerr << "Cost of first tape:" << endl;
  offline_data_used.print_cost();
}


ostream& operator<<(ostream& s,const Program& P)
{
  for (unsigned int i=0; i<P.p.size(); i++)
    { s << i << " :: " << P.p[i] << endl; }
  return s;
}
