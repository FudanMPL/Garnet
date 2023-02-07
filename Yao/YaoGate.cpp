/*
 * YaoGate.cpp
 *
 */

#include "YaoGate.h"
#include "YaoGarbler.h"
#include "YaoEvaluator.h"
#include "BMR/prf.h"
#include "Tools/MMO.h"

YaoFullGate::YaoFullGate(const YaoGarbleWire& out, const YaoGarbleWire& left,
		const YaoGarbleWire& right, Function func)
{
	const Key& delta = YaoGarbler::s().get_delta();
	MMO& mmo = YaoGarbler::s().mmo;
	Key hashes[4];
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			hashes[2 * i + j] = mmo.hash(
					E_input(left.key() ^ (i ? delta : 0),
							right.key() ^ (j ? delta : 0),
							YaoGarbler::s().get_gate_id()));
	garble(out, hashes, left.mask(), right.mask(), func, delta);
#ifdef DEBUG
	cout << "left " << left.mask() << " " << left.key() << " " << (left.key() ^ delta) << endl;
	cout << "right " << right.mask() << " " << right.key() << " " << (right.key() ^ delta) << endl;
	cout << "out " << out.mask() << " " << out.key() << " " << (out.key() ^ delta) << endl;
#endif
}

void YaoFullGate::eval(YaoEvalWire& out, const YaoEvalWire& left, const YaoEvalWire& right)
{
	MMO& mmo = YaoEvaluator::s().mmo;
	Key key = E_input(left.key(), right.key(), YaoEvaluator::s().get_gate_id());
	Key hash = mmo.hash(key);
	eval(out, &hash, left, right);
#ifdef DEBUG
	cout << "external " << left.external() << " " << right.external() << endl;
	cout << "entry " << get_entry(left.external(), right.external()) << endl;
	cout << "out " << out.key() << endl;
#endif
}
