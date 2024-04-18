/*
 * CommonParty.h
 *
 */

#ifndef BMR_COMMONPARTY_H_
#define BMR_COMMONPARTY_H_

#include <vector>
using namespace std;

#include "config.h"
#include "GarbledGate.h"
#include "Register.h"
#include "proto_utils.h"
#include "network/Node.h"
#include "Tools/random.h"
#include "Tools/time-func.h"
#include "GC/Program.h"
#include "Tools/FlexBuffer.h"

#if (defined(DEBUG) || defined(DEBUG_COMM)) && !defined(DEBUG_STEPS)
#define DEBUG_STEPS
#endif

enum SpdzOp
{
	SPDZ_LOAD,
	SPDZ_STORE,
	SPDZ_MAC,
	SPDZ_OP_N,
};

namespace GC
{
template<class T> class Machine;
}

class CommonParty
{
protected:
    friend class Register;

#ifdef N_PARTIES
    const party_id_t _N = N_PARTIES;
#else
    party_id_t _N;
#endif
    int gate_counter, gate_counter2;
    int garbled_tbl_size;

    Timer cpu_timer;
    Timer timers[2];
    Timer timer;

    gf2n_long mac_key;

    LocalBuffer wires;
    ReceivedMsgStore wire_storage;

    template<class T>
    GC::BreakType first_phase(GC::Program& program, GC::Processor<T>& processor,
            GC::Machine<T>& machine);
    template<class T, class U>
    GC::BreakType second_phase(GC::Program& program, GC::Processor<T>& processor,
            GC::Machine<T>& machine, U& dynamic_memory);

public:
    static CommonParty* singleton;
    static CommonParty& s();

    PRNG prng;

    CommonParty();
    virtual ~CommonParty();

#ifdef N_PARTIES
    static int get_n_parties() { return N_PARTIES; }
#else
    static int get_n_parties() { return s()._N; }
#endif

    void check(int n_parties);

    virtual void reset();

    gate_id_t new_gate();
    void next_gate(GarbledGate& gate);
    gate_id_t next_gate(int skip) { return gate_counter2 += skip; }
    size_t get_garbled_tbl_size() { return garbled_tbl_size; }

    gf2n_long get_mac_key() { return mac_key; }
};

class CommonFakeParty : virtual public CommonParty, public NodeUpdatable
{
protected:
    Node* _node;

    mutex global_lock;

public:
    CommonFakeParty();
    virtual ~CommonFakeParty();

    vector<SendBuffer> buffers;

    void init(const char* netmap_file, int id, int n_parties);
    int init(const char* netmap_file, int id);

    SendBuffer& get_buffer(MSG_TYPE type);
};


inline CommonParty& CommonParty::s()
{
    if (singleton)
        return *singleton;
    else
        throw runtime_error("no singleton");
}

#endif /* BMR_COMMONPARTY_H_ */
