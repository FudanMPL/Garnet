/*
 * GarbledGate.cpp
 *
 */

#include "GarbledGate.h"
#include "prf.h"
#include "CommonParty.h"

GarbledGate::~GarbledGate() {
}

void GarbledGate::init_inputs(gate_id_t g, int n_parties)
{
	n_parties = CommonParty::get_n_parties();
	id = g;
	for (unsigned int e = 0; e <= 1; e++) {
		prf_inputs[e].resize(n_parties);
		for (unsigned int j = 0; j < (size_t)n_parties; j++) {
			prf_inputs[e][j] = 0;

			/* fill out this buffer s.t. first 4 bytes are the extension (0/1),
			 * next 4 bytes are gate_id and next 4 bytes are party id.
			 * For the first half we dont need to fill the extension because
			 * it is zero anyway.
			 */

			unsigned int* prf_input_index =
					(unsigned int*) &prf_inputs[e][j]; //easier to refer as integers
			//				printf("e,g,j=%u,%u,%u\n",e,g,j);
			*prf_input_index = e;
			*(prf_input_index + 1) = g;
			*(prf_input_index + 2) = j + 1;
		}
	}
}

void GarbledGate::compute_prfs_outputs(const Register** in_wires, int my_id,
        PRFOutputs& prf_output, gate_id_t g)
{
    int n_parties = CommonParty::get_n_parties();
    init_inputs(g, n_parties);
    for(int w=0; w<=1; w++) {
        for (int b=0; b<=1; b++) {
            const Key& key = in_wires[w]->key(my_id, b);
            __m128i rd_key[15];
            aes_128_schedule((octet*) rd_key, (unsigned char*)&key.r);
#ifdef DEBUG
            cout << "using key " << key << endl;
#endif
            for (int e=0; e<=1; e++) {
                for (int j=1; j<= n_parties; j++) {
                    prf_output[j-1].outputs[w][b][e][0] =
                            aes_128_encrypt(*(__m128i*)input(e, j), (octet*)rd_key);
                }
            }
        }
    }
}

void GarbledGate::compute_prfs_outputs(const Register** in_wires, int my_id,
        SendBuffer& buffer, gate_id_t g)
{
    int n_parties = CommonParty::get_n_parties();
    PRFOutputs prf_output(n_parties);
    compute_prfs_outputs(in_wires, my_id, prf_output, g);
    prf_output.serialize(buffer, my_id, n_parties);
#ifdef DEBUG
	wire_id_t wire_ids[] = { (wire_id_t)in_wires[0]->get_id(), (wire_id_t)in_wires[1]->get_id() };
	prf_output.print_prfs(g, wire_ids, my_id, n_parties);
#endif
}

void PRFOutputs::serialize(SendBuffer& buffer, int my_id, int n_parties)
{
    (void) my_id;
    for (int i = 0; i < n_parties; i++)
        buffer.serialize(tuples[i]);
}

void PRFOutputs::print_prfs(gate_id_t g, wire_id_t* in_wires, party_id_t my_id, int n_parties)
{
	for(int w=0; w<=1; w++) {
		for (int b=0; b<=1; b++) {
			for (int e=0; e<=1; e++) {
				for(party_id_t j=1; j<=(size_t)n_parties; j++) {
					printf("F_k^%d_{%lu,%u}(%d,%lu,%u) = ", my_id, in_wires[w], b, e, g, j);
					Key k = *((Key*)(*this)[j-1].outputs[w][b][e]);
					std::cout << k << std::endl;
				}
			}
		}
	}
}

void GarbledGate::print()
{
	cout << "garbled gate " << id << endl;
	for (int i = 0; i < 4; i++)
	{
		for (size_t j = 0; j < keys[i].size(); j++)
			cout << keys[i][j] << " ";
		cout << endl;
	}
}
