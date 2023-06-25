/*
 * prf.h
 *
 */

#ifndef PROTOCOL_INC_PRF_H_
#define PROTOCOL_INC_PRF_H_

#include "Key.h"

#include "Tools/aes.h"

inline void PRF_chunk(const Key& key, char* input, char* output, int number)
{
	__m128i* in = (__m128i*)input;
	__m128i* out = (__m128i*)output;
	__m128i rd_key[15];
	aes_128_schedule((octet*) rd_key, (unsigned char*)&key.r);
	switch (number)
	{
	case 2:
		ecb_aes_128_encrypt<2>(out, in, (octet*)rd_key);
		break;
	case 3:
		ecb_aes_128_encrypt<3>(out, in, (octet*)rd_key);
		break;
	default:
		for (int i = 0; i < number; i++)
			ecb_aes_128_encrypt<1>(&out[i], &in[i], (octet*)rd_key);
		break;
	}
}

#endif /* PROTOCOL_INC_PRF_H_ */
