/*
 * YaoHalfGate.h
 *
 */

#ifndef YAO_YAOHALFGATE_H_
#define YAO_YAOHALFGATE_H_

#include "BMR/Key.h"
#include "YaoGarbleWire.h"
#include "YaoEvalWire.h"

class YaoHalfGate
{
	Key TG, TE;

public:
	static const int N_EVAL_HASHES = 2;

	static void eval_inputs(Key* output, const Key& left, const Key& right,
			long T);
	static void E_inputs(Key* output, const YaoGarbleWire& left,
			const YaoGarbleWire& right, const Key& left_delta,
			const Key& right_delta, long T);
	static void randomize(YaoGarbleWire&, PRNG&) {}
	static Key garble_public_input(bool value, Key delta)
	{
		return value ? delta : 0;
	}

	YaoHalfGate() {}
	YaoHalfGate(YaoGarbleWire&, const YaoGarbleWire&,
			const YaoGarbleWire&, Function);
	void and_garble(YaoGarbleWire& out, const Key* hashes,
			const YaoGarbleWire& left, const YaoGarbleWire& right, Key delta);
	void garble(const YaoGarbleWire&, const Key*, bool,
			bool, Function, Key);
	void eval(YaoEvalWire&, const YaoEvalWire&,
			const YaoEvalWire&);
	void eval(YaoEvalWire& out, const Key* hashes, const YaoEvalWire& left,
			const YaoEvalWire& right);
};

inline void YaoHalfGate::E_inputs(Key* output, const YaoGarbleWire& left,
		const YaoGarbleWire& right, const Key& left_delta, const Key&, long T)
{
	auto l = left.full_key().doubling(1);
	auto r = right.full_key().doubling(1);
	long j = T << 1;
	output[0] = l ^ j;
	output[1] = output[0] ^ left_delta;
	output[2] = r ^ (j + 1);
	output[3] = output[2] ^ left_delta;
}

inline void YaoHalfGate::and_garble(YaoGarbleWire& out, const Key* hashes,
		const YaoGarbleWire& left, const YaoGarbleWire& right, Key delta)
{
	bool pa = left.mask();
	bool pb = right.mask();
	TG = hashes[0] ^ hashes[1];
	if (pb)
		TG ^= delta;
	Key WG = hashes[0];
	if (pa)
		WG ^= TG;
	TE = hashes[2] ^ hashes[3] ^ left.full_key();
	Key WE = hashes[2];
	if (pb)
		WE ^= TE ^ left.full_key();
	out.set_full_key(WG ^ WE);
	assert(out.mask() == out.full_key().get_signal());
}

inline void YaoHalfGate::eval_inputs(Key* output, const Key& left,
		const Key& right, long T)
{
	long j = T << 1;
	output[0] = left.doubling(1) ^ j;
	output[1] = right.doubling(1) ^ (j + 1);
}

inline void YaoHalfGate::eval(YaoEvalWire& out, const Key* hashes,
		const YaoEvalWire& left, const YaoEvalWire& right)
{
	bool sa = left.external();
	bool sb = right.external();
	Key WG = hashes[0];
	if (sa)
		WG ^= TG;
	Key WE = hashes[1];
	if (sb)
		WE ^= TE ^ left.key();
	out.set(WG ^ WE);
}

#endif /* YAO_YAOHALFGATE_H_ */
