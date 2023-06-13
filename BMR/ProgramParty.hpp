/*
 * ProgramParty.hpp
 *
 */

#ifndef BMR_PROGRAMPARTY_HPP_
#define BMR_PROGRAMPARTY_HPP_

#include "Party.h"

#include "GC/ShareSecret.hpp"

template<class T>
ProgramPartySpec<T>* ProgramPartySpec<T>::singleton = 0;

template<class T>
ProgramPartySpec<T>::ProgramPartySpec() : MC(0)
{
	assert(singleton == 0);
	singleton = this;
}

template<class T>
ProgramPartySpec<T>::~ProgramPartySpec()
{
	if (MC)
		delete MC;
}

template<class T>
void ProgramPartySpec<T>::load(string progname)
{
	program.parse(progname + "-0");
	machine.reset(program, dynamic_memory);
	processor.reset(program);
	prf_machine.reset(program);
	prf_processor.reset(program);
}

template<class T>
void ProgramPartySpec<T>::_check_evaluate()
{
#ifdef DEBUG_REGS
	print_round_regs();
#endif
#ifdef VERBOSE
	cerr << "Online time at evaluation start: " << online_timer.elapsed()
			<< endl;
#endif
	GC::BreakType next = GC::TIME_BREAK;
	while (next == GC::TIME_BREAK)
	{
		load_garbled_circuit();
		next = second_phase(program, processor, machine, dynamic_memory);
	}
#ifdef VERBOSE
	cerr << "Online time at evaluation stop: " << online_timer.elapsed()
			<< endl;
#endif
	if (next == GC::TIME_BREAK)
	{
#ifdef DEBUG_STEPS
		cout << "another round of garbling" << endl;
#endif
	}
	if (next == GC::CLEANING_BREAK)
		return;
	if (next != GC::DONE_BREAK)
	{
#ifdef DEBUG_STEPS
		cout << "another round of evaluation" << endl;
#endif
		start_online_round();
	}
	else
	{
		Timer timer;
		timer.start();
		MC->Check(*P);
#ifdef VERBOSE
		cerr << "Final check took " << timer.elapsed() << endl;
#endif
		done();
		machine.write_memory(N.my_num());
	}
}

template<class T>
void ProgramPartySpec<T>::get_spdz_wire(SpdzOp op, DualWire<T>& spdz_wire)
{
	while (true)
	{
		if (spdz_wires[op].empty())
			throw runtime_error("no SPDZ wires available");
		if (spdz_wires[op].front().done())
			spdz_wires[op].pop_front();
		else
			break;
	}
	spdz_wire.unpack(spdz_wires[op].front(), get_n_parties());
	spdz_counters[op]++;
#ifdef DEBUG_SPDZ_WIRE
	cout << "get SPDZ wire of type " << op << ", " << spdz_wires[op].front().left() << " bytes left" << endl;
	cout << "mask share for " << get_id() << ": " << spdz_wire.mask << endl;
#endif
}

#endif /* BMR_PROGRAMPARTY_HPP_ */
