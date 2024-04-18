/*
 * YaoEvalInput.h
 *
 */

#ifndef YAO_YAOEVALINPUT_H_
#define YAO_YAOEVALINPUT_H_

#include "YaoEvaluator.h"

class YaoEvalInput
{
public:
	YaoEvaluator& evaluator;
	BitVector inputs;
	int i_bit;
	octetStream os;

	YaoEvalInput() :
			evaluator(YaoEvaluator::s())
	{
		inputs.resize(0);
		i_bit = 0;
	}

	void exchange()
	{
		evaluator.ot_ext.extend_correlated(inputs.size(), inputs);
		evaluator.player.receive(os);
	}
};

#endif /* YAO_YAOEVALINPUT_H_ */
