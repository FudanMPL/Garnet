/*
 * YaoHalfGate.cpp
 *
 */

#include "YaoHalfGate.h"
#include "YaoGarbler.h"
#include "YaoEvaluator.h"

YaoHalfGate::YaoHalfGate(YaoGarbleWire& out, const YaoGarbleWire& left,
		const YaoGarbleWire& right, Function function)
{
	for (int i = 0; i < 4; i++)
		assert(function[i] == Function(0x0001)[i]);
	Key labels[4];
	Key hashes[4];
	E_inputs(labels, left, right, YaoGarbler::s().get_delta().doubling(1),
			{}, YaoGarbler::s().counter);
	YaoGarbler::s().mmo.hash<4>(hashes, labels);
	and_garble(out, hashes, left, right, YaoGarbler::s().get_delta());
}

void YaoHalfGate::eval(YaoEvalWire& out, const YaoEvalWire& left,
		const YaoEvalWire& right)
{
	Key hashes[2];
	Key labels[2];
	eval_inputs(labels, left.key(), right.key(), YaoEvaluator::s().counter);
	YaoEvaluator::s().mmo.hash<2>(hashes, labels);
	eval(out, hashes, left, right);
}
