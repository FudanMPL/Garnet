#ifndef _BASE_OT
#define _BASE_OT

/* The OT thread uses the Miracl library, which is not thread safe.
 * Thus all Miracl based code is contained in this one thread so as
 * to avoid locking issues etc.
 *
 * Thus this thread serves all base OTs to all other threads
 */

#include "Networking/Player.h"
#include "Tools/random.h"
#include "Tools/BitVector.h"

// currently always assumes BOTH, i.e. do 2 sets of OT symmetrically,
// use bitwise & to check for role
enum OT_ROLE
{
	RECEIVER = 0x01,
	SENDER = 0x10,
	BOTH = 0x11
};

OT_ROLE INV_ROLE(OT_ROLE role);

const char* role_to_str(OT_ROLE role);
void send_if_ot_sender(TwoPartyPlayer* P, vector<octetStream>& os, OT_ROLE role);
void send_if_ot_receiver(TwoPartyPlayer* P, vector<octetStream>& os, OT_ROLE role);

class BaseOT
{
    static void hash_with_id(BitVector& bits, long id);

public:
	BitVector receiver_inputs;
	vector< array<BitVector, 2> > sender_inputs;
	vector<BitVector> receiver_outputs;
	TwoPartyPlayer* P;
	int nOT, ot_length;
	OT_ROLE ot_role;

	BaseOT(int nOT, int ot_length, TwoPartyPlayer* player, OT_ROLE role=BOTH)
		: P(player), nOT(nOT), ot_length(ot_length), ot_role(role)
	{
		receiver_inputs.resize(nOT);
		sender_inputs.resize(nOT);
		receiver_outputs.resize(nOT);
		G_sender.resize(nOT);
		G_receiver.resize(nOT);

		for (int i = 0; i < nOT; i++)
		{
			sender_inputs[i][0] = BitVector(8 * AES_BLK_SIZE);
			sender_inputs[i][1] = BitVector(8 * AES_BLK_SIZE);
			receiver_outputs[i] = BitVector(8 * AES_BLK_SIZE);
		}
	}

	BaseOT(TwoPartyPlayer* player, OT_ROLE role) :
			BaseOT(128, 128, player, role)
	{
	}

	virtual ~BaseOT() {}

	int length() { return ot_length; }

	void set_receiver_inputs(const BitVector& new_inputs)
	{
		if ((int)new_inputs.size() != nOT)
			throw invalid_length("BaseOT");
		receiver_inputs = new_inputs;
	}

	void set_receiver_inputs(int128 inputs)
	{
		BitVector new_inputs(128);
		for (int i = 0; i < 128; i++)
			new_inputs[i] = (inputs >> i).get_lower() & 1;
		set_receiver_inputs(new_inputs);
	}

	// do the OTs -- generate fresh random choice bits by default
	virtual void exec_base(bool new_receiver_inputs=true);
	// use PRG to get the next ot_length bits
	void set_seeds();
	void extend_length();
	void check();

protected:
	vector< array<PRNG, 2> > G_sender;
	vector<PRNG> G_receiver;

	bool is_sender() { return (bool) (ot_role & SENDER); }
	bool is_receiver() { return (bool) (ot_role & RECEIVER); }

	template<class T, class U>
	void exec_base(bool new_receiver_inputs=true);
};

class FakeOT : public BaseOT
{
public:
   FakeOT(int nOT, int ot_length, TwoPartyPlayer* player, OT_ROLE role=BOTH) :
       BaseOT(nOT, ot_length, player, role) {}
   void exec_base(bool new_receiver_inputs=true);
};

#endif
