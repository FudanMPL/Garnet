/*
 * Key.cpp
 *
 */


#include <string.h>
#include "Key.h"

ostream& operator<<(ostream& o, const Key& key)
{
	return o << key.r;
}

ostream& operator<<(ostream& o, const __m128i& x) {
	o.fill('0');
	o << hex << noshowbase;
	for (int i = 0; i < 2; i++)
	{
		o.width(16);
		o << ((int64_t*)&x)[1-i];
	}
	o << dec;
	return o;
}
