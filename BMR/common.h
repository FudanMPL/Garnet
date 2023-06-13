/*
 * common.h
 *
 */

#ifndef CIRCUIT_INC_COMMON_H_
#define CIRCUIT_INC_COMMON_H_

#include <cstring>
#include <string>
#include <vector>
using namespace std;

#include "Tools/CheckVector.h"

typedef unsigned long wire_id_t;
typedef unsigned long gate_id_t;
typedef unsigned int party_id_t;

class Function {
	bool rep[4];
	int shift(int i) { return 4 * (3 - i); }
public:
	Function() { memset(rep, 0, sizeof(rep)); }
	Function(std::string& func)
	{
		for (int i = 0; i < 4; i++)
			if (func[i] != '0')
				rep[i] = 1;
			else
				rep[i] = 0;
	}
	Function(int int_rep)
	{
		for (int i = 0; i < 4; i++)
			rep[i] = (int_rep << shift(i)) & 1;
	}
	uint8_t operator[](int i) { return rep[i]; }
	bool call(bool left, bool right) { return rep[2 * left + right]; }
};

#endif /* CIRCUIT_INC_COMMON_H_ */
