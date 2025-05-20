#ifndef _DiscreteGauss
#define _DiscreteGauss

/* Class to sample from a Discrete Gauss distribution of 
   standard deviation R
*/

#include <FHE/Generator.h>
#include "Tools/random.h"
#include <vector>
#include <math.h>

class DiscreteGauss
{
  /* This is the bound we use on for the NewHope approximation
   * to a discrete Gaussian with sigma=sqrt(B/2)
   */
  int NewHopeB;

  public:

  void set(double R);

  void pack(octetStream& o) const { o.serialize(NewHopeB); }
  void unpack(octetStream& o) { o.unserialize(NewHopeB); }

  DiscreteGauss(double R) { set(R); }

  // Rely on default copy constructor/assignment
  
  int sample(PRNG& G, int stretch = 1) const;
  double get_R() const { return sqrt(0.5 * NewHopeB); }
  int get_NewHopeB() const { return NewHopeB; }

  bool operator!=(const DiscreteGauss& other) const;
};

template<class T>
class RandomGenerator : public Generator<T>
{
protected:
  mutable PRNG G;

public:
  RandomGenerator(PRNG& G) { this->G.SetSeed(G); }
};

template<class T>
class UniformGenerator : public RandomGenerator<T>
{
  int n_bits;
  bool positive;

public:
  UniformGenerator(PRNG& G, int n_bits, bool positive = true) :
    RandomGenerator<T>(G), n_bits(n_bits), positive(positive) {}
  Generator<T>* clone() const { return new UniformGenerator<T>(*this); }
  void get(T& x) const  { this->G.get(x, n_bits, positive); }
};

template<class T = bigint>
class GaussianGenerator : public RandomGenerator<T>
{
  DiscreteGauss DG;
  int stretch;

public:
  GaussianGenerator(const DiscreteGauss& DG, PRNG& G, int stretch = 1) :
    RandomGenerator<T>(G), DG(DG), stretch(stretch) {}
  Generator<T>* clone() const { return new GaussianGenerator<T>(*this); }
  void get(T& x) const { x = DG.sample(this->G, stretch); }
};

int sample_half(PRNG& G);

template<class T>
class HalfGenerator : public RandomGenerator<T>
{
public:
  HalfGenerator(PRNG& G) :
    RandomGenerator<T>(G) {}
  Generator<T>* clone() const { return new HalfGenerator<T>(*this); }
  void get(T& x) const { x = sample_half(this->G); }
};

#endif
