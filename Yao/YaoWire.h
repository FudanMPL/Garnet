/*
 * YaoWire.h
 *
 */

#ifndef YAO_YAOWIRE_H_
#define YAO_YAOWIRE_H_

#include "BMR/Key.h"
#include "BMR/Register.h"

class YaoWire : public Phase
{
protected:
	Key key_;

public:
	static int threshold(int) { return 1; }

	template<class T>
	static void xors(GC::Processor<T>& processor, const vector<int>& args);
	template<class T>
	static void xors(GC::Processor<T>& processor, const vector<int>& args,
			size_t start, size_t end);

	template<class T>
	static void andm(GC::Processor<T>& processor,
			const BaseInstruction& instruction);

	void XOR(const YaoWire& left, const YaoWire& right)
	{
		key_ = left.key_ ^ right.key_;
	}

	template<class T>
	void other_input(T&, int) {}
};

#endif /* YAO_YAOWIRE_H_ */
