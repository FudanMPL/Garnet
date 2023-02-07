/*
 * msg_types.h
 *
 */

#ifndef PROTOCOL_INC_MSG_TYPES_H_
#define PROTOCOL_INC_MSG_TYPES_H_


#define MESSAGE_TYPES \
	X(TYPE_KEYS) \
	X(TYPE_MASK_INPUTS) \
	X(TYPE_MASK_OUTPUT) \
	X(TYPE_PRF_OUTPUTS) \
	X(TYPE_GARBLED_CIRCUIT) \
	X(TYPE_LAUNCH_ONLINE) \
	X(TYPE_RECEIVED_GC) \
	X(TYPE_SPDZ_WIRES) \
	X(TYPE_DELTA) \
	X(TYPE_NEXT) \
	X(TYPE_DONE) \
	X(TYPE_MAX) \


#define X(NAME) NAME,

typedef enum {
MESSAGE_TYPES
} MSG_TYPE;

#undef X

extern const char* message_type_names[];

#endif /* PROTOCOL_INC_MSG_TYPES_H_ */
