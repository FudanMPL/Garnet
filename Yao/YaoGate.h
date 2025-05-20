/*
 * YaoGate.h
 *
 */

#ifndef YAO_YAOGATE_H_
#define YAO_YAOGATE_H_

#include "config.h"
#include "BMR/Key.h"
#include "YaoGarbleWire.h"
#include "YaoEvalWire.h"
#include "YaoHalfGate.h"

class YaoFullGate
{
	Key entries[2][2];

public:
	static const int N_EVAL_HASHES = 1;

	static Key E_input(const Key& left, const Key& right, long T);
	static void E_inputs(Key* output, const YaoGarbleWire& left,
			const YaoGarbleWire& right,
			const Key& left_delta, const Key& right_delta,
			long T);
	static void eval_inputs(Key* out, const Key& left, const Key& right, long T)
	{
		*out = E_input(left, right, T);
	}
	static void randomize(YaoGarbleWire& out, PRNG& prng)
	{
		out.randomize(prng);
	}
	static Key garble_public_input(bool value, Key)
	{
		return value;
	}

	YaoFullGate() {}
	YaoFullGate(const YaoGarbleWire& out, const YaoGarbleWire& left,
			const YaoGarbleWire& right, Function func);
	void and_garble(const YaoGarbleWire& out, const Key* hashes, const YaoGarbleWire& left,
			const YaoGarbleWire& right, Key delta);
	void garble(const YaoGarbleWire& out, const Key* hashes, bool left_mask,
			bool right_mask, Function func, Key delta);
	void eval(YaoEvalWire& out, const YaoEvalWire& left, const YaoEvalWire& right);
	void eval(YaoEvalWire& out, const Key* hash, const YaoEvalWire& left,
			const YaoEvalWire& right);
	const Key& get_entry(bool left, bool right) { return entries[left][right]; }
};

inline Key YaoFullGate::E_input(const Key& left, const Key& right, long T)
{
	Key res = left.doubling(1) ^ right.doubling(2) ^ T;
#ifdef DEBUG
	cout << "E " << res << ": " << left.doubling(1) << " " << right.doubling(2)
			<< " " << T << endl;
#endif
	return res;
}

inline void YaoFullGate::E_inputs(Key* output, const YaoGarbleWire& left,
		const YaoGarbleWire& right,
		const Key& left_delta, const Key& right_delta, long T)
{
	auto l = left.key().doubling(1);
	auto r = right.key().doubling(2);

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			output[2 * i + j] = l ^ (i ? left_delta : 0) ^ r
					^ (j ? right_delta : 0) ^ T;
}

__attribute__((always_inline))
inline void YaoFullGate::and_garble(const YaoGarbleWire& out, const Key* hashes,
		const YaoGarbleWire& left, const YaoGarbleWire& right, Key delta)
{
	bool left_mask = left.mask();
	bool right_mask = right.mask();
#define XX(L, R, O) \
	for (int left = 0; left < 2; left++) \
		for (int right = 0; right < 2; right++) \
		{ \
			int index = 2 * left + right; \
			Key key = out.key(); \
			if (((left ^ L) & (right ^ R)) ^ O) \
				key += delta; \
			key += hashes[index]; \
			entries[left][right] = key; \
		}
#define Y(L, R) \
	if (out.mask()) \
		XX(L, R, true) \
	else \
		XX(L, R, false)
#define Z(L) \
	if (right_mask) \
		Y(L, true) \
	else \
		Y(L, false)

	if (left_mask) \
		Z(true) \
	else \
		Z(false)
}

inline void YaoFullGate::garble(const YaoGarbleWire& out, const Key* hashes,
		bool left_mask, bool right_mask, Function func, Key delta)
{
	for (int left = 0; left < 2; left++)
		for (int right = 0; right < 2; right++)
		{
			Key key = out.key();
			if (func.call(left ^ left_mask, right ^ right_mask) ^ out.mask())
				key += delta;
#ifdef DEBUG
			cout << "start key " << key << endl;
#endif
			key += hashes[2 * (left) + (right)];
#ifdef DEBUG
			cout << "after left " << key << endl;
#endif
			entries[left][right] = key;
		}
#ifdef DEBUG
	//cout << "counter " << YaoGarbler::s().counter << endl;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			cout << "entry " << i << " " << j << " " << entries[i][j] << endl;
#endif
}

inline void YaoFullGate::eval(YaoEvalWire& out, const Key* hash,
		const YaoEvalWire& left, const YaoEvalWire& right)
{
	Key key = get_entry(left.external(), right.external());
	key -= *hash;
#ifdef DEBUG
	cout << "after left " << key << endl;
#endif
	out.set(key);
}

#endif /* YAO_YAOGATE_H_ */
