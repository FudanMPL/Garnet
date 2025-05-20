/*
 * TrustedParty.cpp
 *
 */

#include "TrustedParty.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <string.h>
#include <iostream>

#include "proto_utils.h"
#include "msg_types.h"
#include "SpdzWire.h"
#include "Protocols/fake-stuff.h"

#include "Register_inline.h"

#include "CommonParty.hpp"
#include "Protocols/fake-stuff.hpp"
#include "BMR/Register.hpp"
#include "GC/Machine.hpp"
#include "GC/Processor.hpp"
#include "GC/Secret.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Program.hpp"
#include "GC/ShareSecret.hpp"
#include "Processor/Instruction.hpp"
#include "Protocols/Share.hpp"

TrustedProgramParty* TrustedProgramParty::singleton = 0;



BaseTrustedParty::BaseTrustedParty()
{
	_num_prf_received = 0;
	_received_gc_received = 0;
	n_received = 0;
	randomfd = open("/dev/urandom", O_RDONLY);
	done_filling = false;
}

BaseTrustedParty::~BaseTrustedParty()
{
    close(randomfd);
}

TrustedProgramParty::TrustedProgramParty(int argc, char** argv) :
		processor(machine),
		random_processor(random_machine)
{
	if (argc < 2)
	{
		cerr << "Usage: " << argv[0] << " <program> [netmap]" << endl;
		exit(1);
	}
	program.parse(string(argv[1]) + "-0");
	processor.reset(program);
	machine.reset(program);
	random_processor.reset(program);
	random_machine.reset(program);
	if (singleton)
		throw runtime_error("there can only be one");
	singleton = this;
	if (argc == 3)
		init(argv[2], 0);
	else
		init("LOOPBACK", 0);
	deltas.resize(_N);
	for (size_t i = 0; i < _N; i++)
	{
		deltas[i] = prng.get_doubleword();
#ifdef DEBUG
		deltas[i] = Key(i + 1, 0);
#endif
		if (deltas[i].get_signal() == 0)
			deltas[i] ^= Key(1);
		cout << "Delta " << i << ": " << deltas[i] << endl;
	}
}

TrustedProgramParty::~TrustedProgramParty()
{
	cout << "Random timer: " << random_timer.elapsed() << endl;
}

void BaseTrustedParty::NodeReady()
{
#ifdef DEBUG_STEPS
	printf("\n\nNode ready \n\n");
#endif
	//sleep(1);
	prepare_randomness();
	send_randomness();
	prf_outputs.resize(get_n_parties());
}

void BaseTrustedParty::prepare_randomness()
{
	msg_keys.resize(_N);
	for (size_t i = 0; i < msg_keys.size(); i++)
	{
		msg_keys[i].clear();
		fill_message_type(msg_keys[i], TYPE_KEYS);
		msg_keys[i].resize(MSG_KEYS_HEADER_SZ);
	}

	done_filling = _fill_keys();
}

void BaseTrustedParty::send_randomness()
{
	for(party_id_t pid=1; pid<=_N; pid++)
	{
//				printf("all keys\n");
//				phex(all_keys, size_of_keys);
		_node->Send(pid, msg_keys[pid - 1]);
//		printf("msg keys\n");
//		phex(msg_keys, msg_keys_size);
		send_input_masks(pid);
	}

	send_output_masks();
}

void TrustedProgramParty::send_input_masks(party_id_t pid)
{
	SendBuffer& buffer = msg_input_masks[pid-1];
#ifdef DEBUG_ROUNDS
	cout << "sending " << buffer.size() << " input masks to " << pid << endl;
#endif
	_node->Send(pid, buffer);
}

void TrustedProgramParty::send_output_masks()
{
#ifdef DEBUG_OUTPUT_MASKS
	cout << "sending " << msg_output_masks.size() - 4 << " output masks" << endl;
#endif
	_node->Broadcast(msg_output_masks);
}

void BaseTrustedParty::NewMessage(int from, ReceivedMsg& msg)
{
	char* message = msg.data();
	int len = msg.size();
	MSG_TYPE message_type;
	msg.unserialize(message_type);
	unique_lock<mutex> locker(global_lock);
	switch(message_type) {
	case TYPE_PRF_OUTPUTS:
	{
#ifdef DEBUG
		cout << "TYPE_PRF_OUTPUTS" << endl;
#endif
		_print_mx.lock();
#ifdef DEBUG2
		printf("got message of len %u from %d\n", len, from);
		phex(message, len);
		cout << "garbled table size " << get_garbled_tbl_size() << endl;
#endif
#ifdef DEBUG_STEPS
		printf("\n Got prfs from %d\n",from);
#endif

		prf_outputs[from-1] = msg;
		_print_mx.unlock();

		if(++_num_prf_received == _N) {
			_num_prf_received = 0;
			_compute_send_garbled_circuit();
		}
		break;
	}
	case TYPE_RECEIVED_GC:
	{
		if(++_received_gc_received == _N) {
			_received_gc_received = 0;
			if (done_filling)
				_launch_online();
			else
				NodeReady();
		}
		break;
	}
	case TYPE_NEXT:
		if (++n_received == _N)
		{
			n_received = 0;
			send_randomness();
		}
		break;
	case TYPE_DONE:
	    if (++n_received == _N)
	        _node->Stop();
	    break;
	default:
		{
		_print_mx.lock();
		printf("got message of len %u from %d\n", len, from);
		printf("UNDEFINED\n");
		printf("got undefined message\n");
		phex(message, len);
		_print_mx.unlock();
		}
		break;
	}

}

void TrustedProgramParty::_launch_online()
{
	_node->Broadcast(get_buffer(TYPE_LAUNCH_ONLINE));
}

void BaseTrustedParty::_compute_send_garbled_circuit()
{
	SendBuffer& buffer = get_buffer(TYPE_GARBLED_CIRCUIT );
	buffer.allocate(get_garbled_tbl_size() * 4 * get_n_parties() * sizeof(Key));
	garble();
	//sending to parties:
#ifdef DEBUG
	cout << "sending garbled circuit" << endl;
#endif
#ifdef DEBUG2
	phex(buffer);
#endif
	_node->Broadcast(buffer);

	//prepare_randomness();
}

void BaseTrustedParty::Start()
{
	_node->Start();
}

void TrustedProgramParty::NodeReady()
{
	for (int i = 0; i < get_n_parties(); i++)
	{
		SendBuffer& buffer = get_buffer(TYPE_DELTA);
		buffer.serialize(deltas[i]);
		_node->Send(i + 1, buffer);
	}
	this->BaseTrustedParty::NodeReady();
}

bool TrustedProgramParty::_fill_keys()
{
	for (int i = 0; i < SPDZ_OP_N; i++)
	{
		spdz_wires[i].clear();
		spdz_wires[i].resize(get_n_parties());
	}
	msg_output_masks = get_buffer(TYPE_MASK_OUTPUT);
	msg_input_masks.resize(get_n_parties());
	for (auto& buffer : msg_input_masks)
	{
		buffer.clear();
		fill_message_type(buffer, TYPE_MASK_INPUTS);
	}
	return GC::DONE_BREAK == first_phase(program, random_processor, random_machine);
}

void TrustedProgramParty::garble()
{
	NoMemory dynamic_memory;
	second_phase(program, processor, machine, dynamic_memory);

	vector< Share<gf2n_long> > tmp;
	make_share(tmp, 1, get_n_parties(), mac_key, prng);
	for (int i = 0; i < get_n_parties(); i++)
		tmp[i].get_mac().pack(spdz_wires[SPDZ_MAC][i]);
	for (int i = 0; i < get_n_parties(); i++)
	{
		for (int j = 0; j < SPDZ_OP_N; j++)
		{
			SendBuffer buffer;
			fill_message_type(buffer, TYPE_SPDZ_WIRES);
			buffer.serialize(j);
			buffer.serialize(spdz_wires[j][i].get_data(), spdz_wires[j][i].get_length());
#ifdef DEBUG_SPDZ_WIRE
			cout << "send " << spdz_wires[j][i].get_length() << "/" << buffer.size()
					<< " bytes for type " << j << " to " << i << endl;
#endif
			_node->Send(i + 1, buffer);
		}
	}
}

void TrustedProgramParty::store_spdz_wire(SpdzOp op, const Register& reg)
{
    make_share(mask_shares, gf2n_long(reg.get_mask()), get_n_parties(),
            gf2n_long(get_mac_key()), prng);
	for (int i = 0; i < get_n_parties(); i++)
	{
		SpdzWire wire;
		wire.mask = mask_shares[i];
		for (int j = 0; j < 2; j++)
		{
			wire.my_keys[j] = reg.keys[j][i];
		}
		wire.pack(spdz_wires[op][i]);
	}
#ifdef DEBUG_SPDZ_WIRE
	cout << "stored SPDZ wire of type " << op << ":" << endl;
	reg.keys.print(reg.get_id());
#endif
}

void TrustedProgramParty::store_wire(const Register& reg)
{
	wires.serialize(reg.mask);
	reg.keys.serialize(wires);
#ifdef DEBUG
	cout << "storing wire" << endl;
	reg.print();
#endif
}

void TrustedProgramParty::load_wire(Register& reg)
{
	wires.unserialize(reg.mask);
	reg.keys.unserialize(wires);
#ifdef DEBUG
	cout << "loading wire" << endl;
	reg.print();
#endif
}
