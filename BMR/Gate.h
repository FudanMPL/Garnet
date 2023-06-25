
#ifndef __GATE_H__
#define __GATE_H__

#include <string>
#include <iostream>
#include <stdio.h>

#include "Key.h"
#include "common.h"

#define NO_LAYER (-1)

class Gate {
public:
	wire_id_t _left;
	wire_id_t _right;
	wire_id_t _out;
	Function _func;

	int _layer;

	Gate() : _left(-1), _right(-1), _out(-1), _layer(-1) {}

	inline void init(wire_id_t left, wire_id_t right, wire_id_t out,std::string func) {
		_left = left;
		_right = right;
		_out = out;
		_func = func;
		_layer = NO_LAYER;
	}
	inline uint8_t func(uint8_t left, uint8_t right) {
		return _func[2*left+right];
	}

	void print(int id) {
		printf ("gate %d: l:%lu, r:%lu, o:%lu, func:%d%d%d%d\n", id, _left, _right, _out, _func[0], _func[1], _func[2], _func[3] );
	}

};

#endif
