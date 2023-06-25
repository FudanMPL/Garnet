/*
 * Checking.h
 *
 */

#ifndef FHEOFFLINE_CHECKING_H_
#define FHEOFFLINE_CHECKING_H_

#include "Networking/Player.h"
#include "Protocols/MAC_Check.h"
#include "Math/Setup.h"
#include "Math/gfpvar.h"

template <class T>
class TripleSacriFactory
{
public:
    vector<array<T, 3>> triples;

    virtual ~TripleSacriFactory() {}
    virtual void get(T& a, T& b, T& c) = 0;
};

template <class T>
class TupleSacriFactory
{
public:
    vector<array<T, 2>> tuples;

    virtual ~TupleSacriFactory() {}
    virtual void get(T& a, T& b) = 0;
};

template <class T>
class SingleSacriFactory
{
public:
    virtual ~SingleSacriFactory() {}
    virtual void get(T& a) = 0;
};

template <class T>
void Triple_Checking(const Player& P, MAC_Check<T>& MC, int nm,
        int output_thread, TripleSacriFactory<Share<T> >& factory,
        bool write_output = true, bool clear = true, string dir = PREP_DIR);
template <class T>
void Inverse_Checking(const Player& P, MAC_Check<T>& MC, int nr,
        int output_thread, TripleSacriFactory<Share<T> >& triple_factory,
        TupleSacriFactory<Share<T> >& inverse_factor,
        bool write_output = true, bool clear = true, string dir = PREP_DIR);
template <class T>
void Square_Checking(const Player& P, MAC_Check<T>& MC, int ns,
        int output_thread, TupleSacriFactory<Share<T> >& factory,
        bool write_output = true, bool clear = true, string dir = PREP_DIR);
void Bit_Checking(const Player& P, MAC_Check<gfp>& MC, int nb,
        int output_thread, TupleSacriFactory<Share<gfp> >& square_factory,
        SingleSacriFactory<Share<gfp> >& bit_factory, bool write_output = true,
        bool clear = true, string dir = PREP_DIR);

template <class T>
inline string file_completion(const T& dummy = {})
{
  (void)dummy;
  return { T::type_char() };
}

#endif /* FHEOFFLINE_CHECKING_H_ */
