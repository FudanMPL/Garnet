/*
 * Checking.cpp
 *
 */

#include "FHE/P2Data.h"

#include "Sacrificing.h"
#include "Producer.h"

#include "Tools/Subroutines.h"

#include "Protocols/mac_key.hpp"

// The number of sacrifices to amortize at one time
#define amortize 512


template <class T>
void Triple_Checking(const Player& P, MAC_Check<T>& MC, int nm,
    int output_thread, TripleSacriFactory< Share<T> >& factory, bool write_output,
    bool clear, string dir)
{
  check_field_size<T>();

  ofstream outf;
  if (write_output)
    open_prep_file<T>(outf, "Triples", P.my_num(), output_thread, false,
        clear, dir);

  T te,t;
  Create_Random(t,P);
  vector<Share<T> > Sh_PO(2*amortize),Sh_Tau(amortize);
  vector<T> PO(2*amortize),Tau(amortize);
  vector<Share<T> > a1(amortize),b1(amortize),c1(amortize);
  vector<Share<T> > a2(amortize),b2(amortize),c2(amortize);
  Share<T> temp;

  // Triple checking
  int left_todo=nm; 
  while (left_todo>0)
    { int this_loop=amortize;
      if (this_loop>left_todo)
        { this_loop=left_todo;
          PO.resize(2*this_loop);
          Sh_PO.resize(2*this_loop);
          Tau.resize(this_loop);
          Sh_Tau.resize(this_loop);
        }
     
      for (int i=0; i<this_loop; i++)
        {
          factory.get(a1[i], b1[i], c1[i]);
          factory.get(a2[i], b2[i], c2[i]);
          Sh_PO[2*i].mul(a1[i],t);
          Sh_PO[2*i].sub(Sh_PO[2*i],a2[i]);
          Sh_PO[2*i+1].sub(b1[i],b2[i]);
	}
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      for (int i=0; i<this_loop; i++)
        { Sh_Tau[i].mul(c1[i],t);
          Sh_Tau[i].sub(Sh_Tau[i],c2[i]);
          temp.mul(a2[i],PO[2*i+1]);
          Sh_Tau[i].sub(Sh_Tau[i],temp);
          temp.mul(b2[i],PO[2*i]);
          Sh_Tau[i].sub(Sh_Tau[i],temp);
          te = (PO[2*i] * PO[2*i+1]);
          Sh_Tau[i].sub(Sh_Tau[i], Share<T>::constant(te,P.my_num(),MC.get_alphai()));
        }
      MC.POpen_Begin(Tau,Sh_Tau,P);
      MC.POpen_End(Tau,Sh_Tau,P);

      for (int i=0; i<this_loop; i++)
        { if (!Tau[i].is_zero())
	    {
              MC.POpen(PO, {a1[i], b1[i], c1[i], a2[i], b2[i], c2[i]}, P);
              for (int j = 0; j < 2; j++)
                {
                  auto prod = PO[3 * j + 1] * PO[3 * j];
                  if (PO[3 * j + 2] != prod)
                    cout << PO[3 * j + 2] << " != " << prod << " = "
                        << PO[3 * j + 1] << " * " << PO[3 * j] << endl;
                }
              throw Offline_Check_Error("Multiplication Triples");
	    }
          if (write_output)
            {
              a1[i].output(outf,false);
              b1[i].output(outf,false);
              c1[i].output(outf,false);
            }
          else
              factory.triples.push_back({{a1[i], b1[i], c1[i]}});
        }

      left_todo-=this_loop;
    }

  if (write_output)
    outf.close();
}


template <class T>
void Inverse_Checking(const Player& P, MAC_Check<T>& MC, int nr,
    int output_thread, TripleSacriFactory<Share<T> >& triple_factory,
    TupleSacriFactory<Share<T> >& inverse_factor, bool write_output,
    bool clear, string dir)
{
  ofstream outf_inv;
  if (write_output)
    open_prep_file<T>(outf_inv, "Inverses", P.my_num(), output_thread, false,
        clear, dir);

  T te,t;
  Create_Random(t,P);
  vector<Share<T> > Sh_PO(2*amortize),Sh_Tau(amortize);
  vector<T> PO(2*amortize),Tau(amortize);
  vector<Share<T> > a1(amortize),b1(amortize),c1(amortize);
  vector<Share<T> > a2(amortize),b2(amortize),c2(amortize);
  Share<T> temp;

  // Inverse checking
  int left_todo=nr;
  while (left_todo>0)
    { int this_loop=amortize;
      if (this_loop>left_todo)
        { this_loop=left_todo;
          PO.resize(2*this_loop);
          Sh_PO.resize(2*this_loop);
          Tau.resize(this_loop);
          Sh_Tau.resize(this_loop);
        }

      for (int i=0; i<this_loop; i++)
        {
          inverse_factor.get(a1[i], b1[i]);
          triple_factory.get(a2[i], b2[i], c2[i]);
          Sh_PO[2*i].mul(a1[i],t);
          Sh_PO[2*i].sub(Sh_PO[2*i],a2[i]);
          Sh_PO[2*i+1].sub(b1[i],b2[i]);
        }
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      auto t_shared = Share<T>::constant(t, P.my_num(), MC.get_alphai());

      for (int i=0; i<this_loop; i++)
        {
          Sh_Tau[i].sub(c2[i],t_shared);
          temp.mul(a2[i],PO[2*i+1]);
          Sh_Tau[i].add(Sh_Tau[i],temp);
          temp.mul(b1[i],PO[2*i]);
          Sh_Tau[i].add(Sh_Tau[i],temp);
        }
      MC.POpen_Begin(Tau,Sh_Tau,P);
      MC.POpen_End(Tau,Sh_Tau,P);

      for (int i=0; i<this_loop; i++)
        { if (!Tau[i].is_zero())
            { throw Offline_Check_Error("Inverses"); }
          if (write_output)
            {
              a1[i].output(outf_inv,false);
              b1[i].output(outf_inv,false);
            }
        }

      left_todo-=this_loop;
    }

  if (write_output)
    {
      outf_inv.close();
    }
}


template <class T>
void Square_Checking(const Player& P, MAC_Check<T>& MC, int ns,
        int output_thread, TupleSacriFactory<Share<T> >& square_factory,
        bool write_output, bool clear, string dir)
{
  ofstream outf_s, outf_b;
  if (write_output)
  {
    open_prep_file<T>(outf_s, "Squares", P.my_num(), output_thread, false, clear, dir);
  }

  T te,t,t2;
  Create_Random(t,P);
  t2 = t * t;
  vector<Share<T> > Sh_PO(amortize);
  vector<T> PO(amortize);
  vector<Share<T> > f(amortize),h(amortize),a(amortize),b(amortize);
  Share<T>  temp;

  // Do the square checking
  int left_todo=ns;
  square_factory.tuples.clear();
   while (left_todo>0)
    { int this_loop=amortize;
      if (this_loop>left_todo)
        { this_loop=left_todo;
          PO.resize(this_loop);
          Sh_PO.resize(this_loop);
        }

      for (int i=0; i<this_loop; i++)
        {
          square_factory.get(f[i], h[i]);
          square_factory.get(a[i], b[i]);

          Sh_PO[i].mul(a[i],t);
          Sh_PO[i].sub(Sh_PO[i],f[i]);
        }
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      for (int i=0; i<this_loop; i++)
        { Sh_PO[i].mul(b[i],t2);
          Sh_PO[i].sub(Sh_PO[i],h[i]);
          temp.mul(a[i],t);
          temp.add(temp,f[i]);
          temp.mul(temp,PO[i]);
          Sh_PO[i].sub(Sh_PO[i],temp);
        }
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      for (int i=0; i<this_loop; i++)
        { if (!PO[i].is_zero())
            { throw Offline_Check_Error("Squares"); }
          if (write_output)
            {
              a[i].output(outf_s,false); b[i].output(outf_s,false);
            }
          else
              square_factory.tuples.push_back({{a[i], b[i]}});
        }
      left_todo-=this_loop;
    }
  outf_s.close(); 
}

void Bit_Checking(const Player& P, MAC_Check<gfp>& MC, int nb,
        int output_thread, TupleSacriFactory<Share<gfp> >& square_factory,
        SingleSacriFactory<Share<gfp> >& bit_factory, bool write_output,
        bool clear, string dir)
{
  gfp dummy;
  ofstream outf_b;
  if (write_output)
    open_prep_file<gfp>(outf_b, "Bits", P.my_num(), output_thread, false, clear,
        dir);

  gfp te,t,t2;
  Create_Random(t,P);
  t2 = t * t;
  vector<Share<gfp> > Sh_PO(amortize);
  vector<gfp> PO(amortize);
  vector<Share<gfp> > f(amortize),h(amortize),a(amortize),b(amortize);
  Share<gfp>  temp;

  // Do the bits checking
  PO.resize(amortize);
  Sh_PO.resize(amortize);
  int left_todo=nb;
  while (left_todo>0)
    { int this_loop=amortize;
      if (this_loop>left_todo)
	{ this_loop=left_todo;
          PO.resize(this_loop);
          Sh_PO.resize(this_loop);
	}

      for (int i=0; i<this_loop; i++)
        {
          square_factory.get(f[i], h[i]);
          bit_factory.get(a[i]);

          Sh_PO[i].mul(a[i],t);
          Sh_PO[i].sub(Sh_PO[i],f[i]);
	}
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      for (int i=0; i<this_loop; i++)
        { Sh_PO[i].mul(a[i],t2);
          Sh_PO[i].sub(Sh_PO[i],h[i]);
          temp.mul(a[i],t);
          temp.add(temp,f[i]);
          temp.mul(temp,PO[i]);
          Sh_PO[i].sub(Sh_PO[i],temp);
        }
      MC.POpen_Begin(PO,Sh_PO,P);
      MC.POpen_End(PO,Sh_PO,P);

      for (int i=0; i<this_loop; i++)
        { if (!PO[i].is_zero())
            { throw Offline_Check_Error("Bits"); }
          if (write_output)
            a[i].output(outf_b,false);
	}

      left_todo-=this_loop;
    }
  outf_b.close();
}


template void Triple_Checking(const Player& P, MAC_Check<gfp>& MC, int nm,
        int output_thread, TripleSacriFactory<Share<gfp> >& factory,
        bool write_output, bool clear, string dir);
template void Triple_Checking(const Player& P, MAC_Check<gf2n_short>& MC,
        int nm, int output_thread,
        TripleSacriFactory<Share<gf2n_short> >& factory, bool write_output,
        bool clear, string dir);

template void Square_Checking(const Player& P, MAC_Check<gfp>& MC, int ns,
        int output_thread, TupleSacriFactory<Share<gfp> >& square_factory,
        bool write_output, bool clear, string dir);
template void Square_Checking(const Player& P, MAC_Check<gf2n_short>& MC,
        int ns, int output_thread,
        TupleSacriFactory<Share<gf2n_short> >& square_factory,
        bool write_output, bool clear, string dir);

template void Inverse_Checking(const Player& P, MAC_Check<gfp>& MC, int nr,
        int output_thread, TripleSacriFactory<Share<gfp> >& triple_factory,
        TupleSacriFactory<Share<gfp> >& inverse_factor, bool write_output,
        bool clear, string dir);
template void Inverse_Checking(const Player& P, MAC_Check<gf2n_short>& MC, int nr,
        int output_thread, TripleSacriFactory<Share<gf2n_short> >& triple_factory,
        TupleSacriFactory<Share<gf2n_short> >& inverse_factor, bool write_output,
        bool clear, string dir);
