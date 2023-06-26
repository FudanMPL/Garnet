
#ifndef _OTVOLE
#define _OTVOLE

#include "Math/Z2k.h"
#include "OTExtension.h"
#include "Row.h"
#include "config.h"

using namespace std;

template <class T>
class OTVoleBase : public OTExtension
{
public:
	const int S;

	OTVoleBase(int S,
	                TwoPartyPlayer* player,
	                OT_ROLE role=BOTH,
	                bool passive=false)
	    : OTExtension(player, role, passive),
	        S(S),
	        corr_prime(),
	        t0(S),
	        t1(S),
	        u(S),
	        t(S),
	        a(S) {
	            // need to flip roles for OT extension init, reset to original role here
	            this->ot_role = role;
	            local_prng.ReSeed();
	        }

	    virtual ~OTVoleBase() {}

	    void evaluate(vector<T>& output, const vector<T>& newReceiverInput);

	    void evaluate(vector<T>& output, int nValues, const BitVector& newReceiverInput);

	    virtual int n_challenges() { return S; }
	    virtual int get_challenge(PRNG&, int i) { return i; }

	protected:

		// Sender fields
		Row<T> corr_prime;
		vector<Row<T>> t0, t1;
		// Receiver fields
		vector<Row<T>> u, t, a;
		// Both
		PRNG local_prng;

	    vector<octetStream> oss;

	    virtual void consistency_check (vector<octetStream>& os);

	    void set_coeffs(__m128i* coefficients, PRNG& G, int num_elements) const;

	    template<class U>
	    void hash_row(octetStream& os, const U& row, const __m128i* coefficients);
	    template<class U>
	    void hash_row(octet* hash, const U& row, const __m128i* coefficients);
	    template<class U>
	    void hash_row(__m128i res[2], const U& row, const __m128i* coefficients);

	    static void add_mul(__m128i res[2], __m128i a, __m128i b);
};

template <class T>
class OTVole : public OTVoleBase<T>
{

public:
    OTVole(int S,
                TwoPartyPlayer* player,
                OT_ROLE role=BOTH,
                bool passive=false)
    : OTVoleBase<T>(S, player, role, passive) {
    }

    int n_challenges() { return NUM_VOLE_CHALLENGES; }
    int get_challenge(PRNG& G, int) { return G.get_uint(this->S); }
};

#endif
