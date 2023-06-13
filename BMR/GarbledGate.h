/*
 * GarbledGate.h
 *
 */

#ifndef BMR_PRIME_CIRCUIT_BMR_INC_GARBLEDGATE_H_
#define BMR_PRIME_CIRCUIT_BMR_INC_GARBLEDGATE_H_

#include "Register.h"
#include "common.h"

struct PRFTuple {
    Key outputs[2][2][2][1];
    // i = 0..3
    Key for_garbling(int i)
    {
       int a = i / 2;
       int b = i % 2;
       return outputs[0][a][b][0] ^ outputs[1][b][a][0];
    }
};

/*
	 *	Total of n*(G*2*2*2*n+4) = n*(8*G*n+RESERVE_FOR_MSG_TYPE) = n*(PRFS_PER_PARTY+RESERVE_FOR_MSG_TYPE)
	 *
	 *	For every party i={1...n}
	 *		<Message_type> = saves us copying memory to another location - only for Party.
	 *		For every gate g={1...G}
	 *			For input wires x={left,right}
	 *				For b={0,1} (0-key/1-key)
	 *					For e={0,1} (extension)
	 *						for every party j={1...n}
	 *							F_{k^i_{x,b}}(e,j,g,)
	 */
struct PRFOutputs {
#ifdef MAX_N_PARTIES
	PRFTuple tuples[MAX_N_PARTIES];
	PRFOutputs(int n_parties) { (void)n_parties; }
#else
	vector<PRFTuple> tuples;

	PRFOutputs(int n_parties) : tuples(n_parties) {}
#endif
	PRFTuple& operator[](int i) { return tuples[i]; }
	void serialize(SendBuffer& buffer, int my_id, int n_parties);
	void print_prfs(gate_id_t g, wire_id_t* in_wires, party_id_t my_id, int n_parties);
};

class GarbledGate : public KeyTuple<4> {
    /*  will be allocated 4*n keys;
     * (n keys for each of A,B,C,D entries):
     *   A1, A2, ... , An
     *   B1, B2, ... , Bn
     *   C1, C2, ... , Cn
     *   D1, D2, ... , Dn
     */

public:
	KeyVector prf_inputs[2]; /*
	   	   * These are all possible inputs to the prf,
	   	   * This is not efficient in terms of storage but increase
	   	   * performance in terms of speed since no need to generate
	   	   * (new allocation plus filling) those inputs every time
	   	   * we compute the prf.
	   	   * Structure:
	   	   * 	- Total of G*n*2 inputs. (i.e. for every gate g, for every
	   	   * 	party j and for every extension e (from {0,1}).
	   	   * 	- We want to be able to set key once and use it to encrypt
	   	   * 	several inputs, so we want those inputs to be adjacent in
	   	   * 	memory in order to save time of building the block of
	   	   * 	inputs. So, the structure of the inputs is as follows:
	   	   * 		- First half of inputs (G*n inputs) are:
	   	   * 			- For every gate g=1,...,G we store the inputs:
	   	   * 				(0||g||1),(0||g||2),...,(0||../circuit_bmr/inc/GarbledGate.h:43:19: error: ‘gate_id_t’ has not been declared
	   	   * 				g||n)
	   	   * 		- Second half of the inputs are:
	   	   * 			- For every gate g=1,...,G we store the inputs:
	   	   * 				(1||g||1),(1||g||2),...,(1||g||n)
	   	   */

	gate_id_t id;

	GarbledGate(int n_parties) : KeyTuple<4>(n_parties), id(-1) {}
	virtual ~GarbledGate();

	void init_inputs(gate_id_t g, int n_parties);

    char* input(int e, party_id_t j) { return (char*)&prf_inputs[e][j-1]; }

    void compute_prfs_outputs(const Register** in_wires, int my_id, SendBuffer& buffer, gate_id_t g);
    void compute_prfs_outputs(const Register** in_wires, int my_id, PRFOutputs& outputs, gate_id_t g);
    void print();
};

#endif /* BMR_PRIME_CIRCUIT_BMR_INC_GARBLEDGATE_H_ */
