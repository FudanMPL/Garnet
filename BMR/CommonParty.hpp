/*
 * CommonParty.hpp
 *
 */

#ifndef BMR_COMMONPARTY_HPP_
#define BMR_COMMONPARTY_HPP_

#include "CommonParty.h"

template <class T>
GC::BreakType CommonParty::first_phase(GC::Program& program,
		GC::Processor<T>& processor, GC::Machine<T>& machine)
{
	(void)machine;
	timers[0].start();
	reset();
	wires.clear();
	NoMemory dynamic_memory;
	GC::BreakType next;
	try
	{
		next = program.execute(processor, dynamic_memory);
	}
	catch (needs_cleaning& e)
	{
		next = GC::CLEANING_BREAK;
		processor.PC--;
	}
#ifdef DEBUG_ROUNDS
	cout << "finished first phase at pc " << processor.PC
			<< " reason " << next << endl;
#endif
	timers[0].stop();
#ifdef VERBOSE
	cerr << "First round time: " << timers[0].elapsed() << " / "
			<< timer.elapsed() << endl;
#endif
#ifdef DEBUG_WIRES
	cout << "Storing wires with " << 1e-9 * wires.size() << " GB on disk" << endl;
#endif
	wire_storage.push(wires);
	return next;
}

template<class T, class U>
GC::BreakType CommonParty::second_phase(GC::Program& program,
		GC::Processor<T>& processor, GC::Machine<T>& machine,
		U& dynamic_memory)
{
	(void)machine;
	wire_storage.pop(wires);
	wires.reset_head();
	timers[1].start();
	GC::BreakType next = GC::TIME_BREAK;
	try
	{
		next = program.execute(processor, dynamic_memory);
	}
	catch (needs_cleaning& e)
	{
		next = GC::CLEANING_BREAK;
        processor.PC--;
	}
#ifdef DEBUG_ROUNDS
	cout << "finished second phase at " << processor.PC
			<< " reason " << next << endl;
#endif
	timers[1].stop();
//	cout << "Second round time: " << timers[1].elapsed() << ", ";
//	cout << "total time: " << timer.elapsed() << endl;
	if (false)
		return GC::CAP_BREAK;
	else
		return next;
}

#endif /* BMR_COMMONPARTY_HPP_ */
