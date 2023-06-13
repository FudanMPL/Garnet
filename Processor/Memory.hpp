#include "Processor/Memory.h"
#include "Processor/Instruction.h"

#include <fstream>

template<class T>
void Memory<T>::minimum_size(RegType secret_type, RegType clear_type,
    const Program &program, const string& threadname)
{
  (void) threadname;
  MS.minimum_size(program.direct_mem(secret_type));
  MC.minimum_size(program.direct_mem(clear_type));
}

template<class T>
void MemoryPart<T>::minimum_size(size_t size)
{
  try
  {
      if (size > this->size())
          this->resize(size);
#ifdef DEBUG_MEMORY_SIZE
      cerr << T::type_string() << " memory has now size " << this->size() << endl;
#endif
  }
  catch (bad_alloc&)
  {
      throw insufficient_memory(size, T::type_string());
  }
}

template<class T>
ostream& operator<<(ostream& s,const Memory<T>& M)
{
  s << M.MS.size() << endl;
  s << M.MC.size() << endl;

#ifdef OUTPUT_HUMAN_READABLE_MEMORY
  for (unsigned int i=0; i<M.MS.size(); i++)
    { M.MS[i].output(s,true); s << endl; }
  s << endl;

  for (unsigned int i=0; i<M.MC.size(); i++)
    {  M.MC[i].output(s,true); s << endl; }
  s << endl;
#else
  for (unsigned int i=0; i<M.MS.size(); i++)
    { M.MS[i].output(s,false); }

  for (unsigned int i=0; i<M.MC.size(); i++)
    { M.MC[i].output(s,false); }
#endif

  return s;
}


template<class T>
istream& operator>>(istream& s,Memory<T>& M)
{
  int len;

  s >> len;  
  M.MS.minimum_size(len);
  s >> len;
  M.MC.minimum_size(len);
  s.seekg(1, istream::cur);

  for (unsigned int i=0; i<M.MS.size(); i++)
    { M.MS[i].input(s,false);  }

  for (unsigned int i=0; i<M.MC.size(); i++)
    { M.MC[i].input(s,false); }

  return s;
}
