/*
 * FlexBuffer.h
 *
 */

#ifndef TOOLS_FLEXBUFFER_H_
#define TOOLS_FLEXBUFFER_H_

#include "Tools/avx_memcpy.h"
#include "Tools/time-func.h"
#include "Tools/octetStream.h"
#include <stdio.h>
#include <stdexcept>
#include <deque>
#include <iostream>
using namespace std;

class FlexBuffer
{
	friend class octetStream;

protected:
    char* buf, *ptr;
	size_t len, max_len;
	void del();
	void reset() { buf = ptr = 0; len = max_len = 0; }
public:
	FlexBuffer() : buf(0), ptr(0), len(0), max_len(0) {}
	FlexBuffer(const FlexBuffer&);
	~FlexBuffer() { del(); }
	void operator=(FlexBuffer& msg);
	void operator=(octetStream& os);
	char* data() { return buf; }
	const char* data() const { return buf; }
	size_t size() const { return len; }
	size_t capacity() const { return max_len; }
};

class ReceivedMsg : public virtual FlexBuffer
{
	friend class ReceivedMsgStore;

public:
	void operator=(FlexBuffer& msg) { FlexBuffer::operator=(msg); }
	void reset_head() { ptr = buf; }
	void resize(size_t new_len);
	void unserialize(void* output, size_t size);
	template <class T>
	void unserialize(T& output);
	template <class T>
	void nonavx_unserialize(T& output);
	char* consume(size_t l) { check_buffer(l); char* res = ptr; ptr += l; return res; }
	char pop_front() { check_buffer(1); return *ptr++; }
	size_t left() { return len - (ptr - buf); }
	void check_buffer(size_t size);
};

class SendBuffer : public virtual FlexBuffer
{
public:
	void operator=(FlexBuffer& msg) { FlexBuffer::operator=(msg); }
	char* end() { return buf + len; }
	void skip(size_t n) { len += n; }
	void resize(size_t new_len);
	void resize_copy(size_t new_max_len);
	void clear() { len = 0; }
	const char& operator[](size_t i) { return data()[i]; }
	void push_back(char c) { serialize(c); }
	template <class T>
	void serialize(const T& source) { serialize(&source, sizeof(T)); }
	void serialize(const void* source, size_t size);
	void allocate(size_t size);
	char* allocate_and_skip(size_t size);
	template <class T>
	void serialize_no_allocate(const T& source);
	void serialize_no_allocate(const void* source, size_t size);
	void send(int socket_num);
};

class LocalBuffer : public ReceivedMsg, public SendBuffer
{
};

class ReceivedMsgStore
{
	static const int N = 1;
	ReceivedMsg mem[N];
	int start, mem_size;
	deque<string> files;
	size_t total_size;
	Timer push_timer, pop_timer;
public:
	ReceivedMsgStore() : start(0), mem_size(0), total_size(0) {}
	~ReceivedMsgStore();
	void push(ReceivedMsg& msg);
	void push_and_clear(LocalBuffer& msg) { push(msg); msg.clear(); }
	bool pop(ReceivedMsg& msg);
	bool empty() { return mem_size == 0 and files.empty(); }
};

inline FlexBuffer::FlexBuffer(const FlexBuffer& msg)
{
	if (msg.buf)
		throw runtime_error("can only copy empty buffers");
	reset();
}

inline void FlexBuffer::operator=(FlexBuffer& msg)
{
    if (this != &msg)
    {
        del();
        buf = msg.buf;
        ptr = msg.ptr;
        len = msg.len;
        max_len = msg.max_len;
#ifdef DEBUG_FLEXBUF
        cout << "moved " << (void*)buf << " " << (void*)msg.buf << " from " << &msg << " to " << this << endl;
#endif
        msg.reset();
    }
}

inline void FlexBuffer::operator=(octetStream& os)
{
	del();
	buf = (char*)os.get_data();
	ptr = (char*)os.get_data() + os.get_ptr();
	len = os.get_length();
	max_len = os.get_max_length();
	os.reset();
}

inline void ReceivedMsg::resize(size_t new_len)
{
	if (new_len > max_len)
	{
		del();
		max_len = new_len;
		buf = new char[max_len];
#ifdef DEBUG_FLEXBUF
		cout << "allocated " << (void*)buf << " for " << this << endl;
#endif
	}
	len = new_len;
	ptr = buf;
}

inline void FlexBuffer::del()
{
#ifdef DEBUG_FLEXBUF
	printf("delete 0x%x for 0x%x\n", buf, this);
#endif
	if (buf)
		delete[] buf;
	reset();
}

inline void ReceivedMsg::unserialize(void* output, size_t size)
{
	check_buffer(size);
	avx_memcpy(output, ptr, size);
	ptr += size;
}

template<class T>
inline void ReceivedMsg::unserialize(T& output)
{
	unserialize(&output, sizeof(T));
}

template<class T>
inline void ReceivedMsg::nonavx_unserialize(T& output)
{
	check_buffer(sizeof(T));
	memcpy(&output, ptr, sizeof(T));
	ptr += sizeof(T);
}

inline void SendBuffer::resize(size_t new_size)
{
	while (size() < new_size)
		push_back(0);
}

inline void SendBuffer::resize_copy(size_t new_max_len)
{
	char* old = buf;
	max_len = new_max_len;
	buf = new char[max_len];
	if (old)
	{
		avx_memcpy(buf, old, len);
		delete[] old;
	}
	ptr = buf + (ptr - old);
}

inline void SendBuffer::serialize(const void* source, size_t size)
{
	allocate(size);
	serialize_no_allocate(source, size);
}

inline void SendBuffer::allocate(size_t size)
{
	if (len + size > max_len)
		resize_copy(2 * (len + size));
}

template<class T>
inline void SendBuffer::serialize_no_allocate(const T& source)
{
	serialize_no_allocate(&source, sizeof(T));
}

inline char* SendBuffer::allocate_and_skip(size_t size)
{
	allocate(size);
	char* res = end();
	skip(size);
	return res;
}

inline void SendBuffer::serialize_no_allocate(const void* source, size_t size)
{
	avx_memcpy(buf + len, source, size);
	len += size;
}

inline void ReceivedMsg::check_buffer(size_t size)
{
	(void)size;
#ifdef CHECK_BUFFER
	if (ptr + size > buf + len)
		throw overflow_error("not enough data in buffer");
#endif
}

#endif /* TOOLS_FLEXBUFFER_H_ */
