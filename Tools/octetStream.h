#ifndef _octetStream
#define _octetStream


/* This class creates a stream of data and adds stuff onto it.
 * This is used to pack and unpack stuff which is sent over the
 * network
 *
 * Unlike SPDZ-1.0 this class ONLY deals with native types
 * For our types we assume pack/unpack operations defined within
 * that class. This is to make sure this class is relatively independent
 * of the rest of the application; and so can be re-used.
 */

// 32-bit length fields for compatibility
#ifndef LENGTH_SIZE
#define LENGTH_SIZE 4
#endif

#include "Networking/data.h"
#include "Networking/sockets.h"
#include "Tools/avx_memcpy.h"

#include <string.h>
#include <vector>
#include <array>
#include <stdio.h>
#include <iostream>
#include <assert.h>

#include <sodium.h>
 
using namespace std;

class bigint;
class FlexBuffer;

/**
 * Buffer for network communication with a pointer for sequential reading.
 * When sent over the network or stored in a file, the length is prefixed
 * as eight bytes in little-endian order.
 */
class octetStream
{
  friend class FlexBuffer;
  template<class T> friend class Exchanger;

  size_t len,mxlen,ptr;  // len is the "write head", ptr is the "read head"
  octet *data;

  void reset();

  public:

  /// Increase allocation if needed
  void resize(size_t l);
  void resize_precise(size_t l);
  void resize_min(size_t l);
  void reserve(size_t l);
  /// Free memory
  void clear();

  void assign(const octetStream& os);

  octetStream() : len(0), mxlen(0), ptr(0), data(0) {}
  /// Initial allocation
  octetStream(size_t maxlen);
  /// Initial buffer
  octetStream(size_t len, const octet* source);
  /// Initial buffer
  octetStream(const string& other);
  octetStream(FlexBuffer& buffer);
  octetStream(const octetStream& os);
  octetStream& operator=(const octetStream& os)
    { if (this!=&os) { assign(os); }
      return *this;
    }
  ~octetStream() { if(data) delete[] data; }
  
  /// Number of bytes already read
  size_t get_ptr() const     { return ptr; }
  /// Length
  size_t get_length() const  { return len; }
  /// Length including size tag
  size_t get_total_length() const  { return len + sizeof(len); }
  /// Allocation
  size_t get_max_length() const { return mxlen; }
  /// Data pointer
  octet* get_data() const { return data; }
  /// Read pointer
  octet* get_data_ptr() const { return data + ptr; }

  /// Whether done reading
  bool done() const 	  { return ptr == len; }
  /// Whether empty
  bool empty() const 	  { return len == 0; }
  /// Bytes left to read
  size_t left() const 	  { return len - ptr; }

  /// Convert to string
  string str() const;

  /// Hash content
  octetStream hash()   const;
  // output must have length at least HASH_SIZE
  void hash(octetStream& output)   const;
  // The following produces a check sum for debugging purposes
  bigint check_sum(int req_bytes=crypto_hash_BYTES)       const;

  /// Append other buffer
  void concat(const octetStream& os);

  /// Reset reading
  void reset_read_head()  { ptr=0; }
  /// Set length to zero but keep allocation
  void reset_write_head() { len=0; ptr=0; }

  // Move len back num
  void rewind_write_head(size_t num) { len-=num; }

  bool equals(const octetStream& a) const;
  /// Equality test
  bool operator==(const octetStream& a) const { return equals(a); }
  bool operator!=(const octetStream& a) const { return not equals(a); }

  /// Append ``num`` random bytes
  void append_random(size_t num);

  /// Append ``l`` bytes from ``x``
  void append(const octet* x,const size_t l);
  octet* append(const size_t l);
  void append_no_resize(const octet* x,const size_t l);
  /// Read ``l`` bytes to ``x``
  void consume(octet* x,const size_t l);
  // Return pointer to next l octets and advance pointer
  octet* consume(size_t l);

  /* Now store and restore different types of data (with padding for decoding) */

  void store_bytes(octet* x, const size_t l); //not really "bytes"...
  void get_bytes(octet* ans, size_t& l);      //Assumes enough space in ans

  /// Append 4-byte integer
  void store(unsigned int a) { store_int(a, 4); }
  /// Append 4-byte integer
  void store(int a);
  /// Read 4-byte integer
  void get(unsigned int& a) { a = get_int(4); }
  /// Read 4-byte integer
  void get(int& a);

  /// Append 8-byte integer
  void store(size_t a) { store_int(a, 8); }
  /// Read 8-byte integer
  void get(size_t& a) { a = get_int(8); }

  /// Append integer of ``n_bytes`` bytes
  void store_int(size_t a, int n_bytes);
  /// Read integer of ``n_bytes`` bytes
  size_t get_int(int n_bytes);

  /// Append integer of ``N_BYTES`` bytes
  template<int N_BYTES>
  void store_int(size_t a);
  /// Read integer of ``N_BYTES`` bytes
  template<int N_BYTES>
  size_t get_int();

  /// Append big integer
  void store(const bigint& x);
  /// Read big integer
  void get(bigint& ans);

  /// Append instance of type implementing ``pack``
  template<class T>
  void store(const T& x);

  /// Read instance of type implementing ``unpack``
  template<class T>
  T get();
  /// Read instance of type implementing ``unpack``
  template<class T>
  void get(T& ans);

  // works for all statically allocated types
  template <class T>
  void serialize(const T& x) { append((octet*)&x, sizeof(x)); }
  template <class T>
  void unserialize(T& x) { consume((octet*)&x, sizeof(x)); }

  /// Append vector of type implementing ``pack``
  template <class T>
  void store(const vector<T>& v);
  /// Read vector of type implementing ``unpack``
  /// @param v results
  /// @param init initialization if required
  template <class T>
  void get(vector<T>& v, const T& init = {});
  /// Read vector of type implementing ``unpack``
  /// if vector already has the right size
  template <class T>
  void get_no_resize(vector<T>& v);

  template <class T, size_t L>
  void store(const array<T, L>& v);
  template <class T, size_t L>
  void get(array<T, L>& v);

  /// Read ``l`` bytes into separate buffer
  void consume(octetStream& s,size_t l)
    { s.resize(l);
      consume(s.data,l);
      s.len=l;
    }

  /// Append string
  void store(const string& str);
  /// Read string
  void get(string& str);

  /// Send on ``socket_num``
  template<class T>
  void Send(T socket_num) const;
  /// Receive on ``socket_num``, overwriting current content
  template<class T>
  void Receive(T socket_num);

  /// Input from stream, overwriting current content
  void input(istream& s);
  /// Output to stream
  void output(ostream& s);

  /// Send on ``socket_num`` while receiving on ``receiving_socket``,
  /// overwriting current content
  template<class T>
  void exchange(T send_socket, T receive_socket) { exchange(send_socket, receive_socket, *this); }
  /// Send this buffer on ``socket_num`` while receiving
  /// to ``receive_stream`` on ``receiving_socket``
  template<class T>
  void exchange(T send_socket, T receive_socket, octetStream& receive_stream) const;

  friend ostream& operator<<(ostream& s,const octetStream& o);
  friend class PRNG;
};

class Player;

class octetStreams : public vector<octetStream>
{
public:
  octetStreams() {}
  octetStreams(const Player& P);

  void reset(const Player& P);
};


inline void octetStream::resize(size_t l)
{
  if (l<mxlen) { return; }
  l=2*l;      // Overcompensate in the resize to avoid calling this a lot
  resize_precise(l);
}


inline void octetStream::resize_precise(size_t l)
{
  if (l == mxlen)
    return;

  octet* nd=new octet[l];
  if (data)
    {
      memcpy(nd, data, min(len, l) * sizeof(octet));
      delete[] data;
    }
  data=nd;
  mxlen=l;
}

inline void octetStream::resize_min(size_t l)
{
  if (l > mxlen)
    resize_precise(l);
}

inline void octetStream::reserve(size_t l)
{
  if (len + l > mxlen)
    resize_precise(len + l);
}

inline octet* octetStream::append(const size_t l)
{
  if (len+l>mxlen)
    resize(len+l);
  octet* res = data + len;
  len+=l;
  return res;
}

inline void octetStream::append(const octet* x, const size_t l)
{
  avx_memcpy(append(l), x, l*sizeof(octet));
}

inline void octetStream::append_no_resize(const octet* x, const size_t l)
{
  avx_memcpy(data+len,x,l*sizeof(octet));
  len+=l;
}

inline octet* octetStream::consume(size_t l)
{
  if(ptr + l > len)
    throw runtime_error("insufficient data");
  octet* res = data + ptr;
  ptr += l;
  return res;
}

inline void octetStream::consume(octet* x,const size_t l)
{
  avx_memcpy(x, consume(l), l * sizeof(octet));
}

inline void octetStream::store_int(size_t l, int n_bytes)
{
  resize(len+n_bytes);
  encode_length(data+len,l,n_bytes);
  len+=n_bytes;
}

inline size_t octetStream::get_int(int n_bytes)
{
  return decode_length(consume(n_bytes), n_bytes);
}

template<int N_BYTES>
inline void octetStream::store_int(size_t l)
{
  assert(N_BYTES <= 8);
  resize(len+N_BYTES);
  uint64_t tmp = htole64(l);
  memcpy(data + len, &tmp, N_BYTES);
  len+=N_BYTES;
}

template<int N_BYTES>
inline size_t octetStream::get_int()
{
  assert(N_BYTES <= 8);
  size_t tmp = 0;
  memcpy(&tmp, consume(N_BYTES), N_BYTES);
  return le64toh(tmp);
}


template<class T>
inline void octetStream::Send(T socket_num) const
{
  send(socket_num,len,LENGTH_SIZE);
  send(socket_num,data,len);
}


template<class T>
inline void octetStream::Receive(T socket_num)
{
  size_t nlen=0;
  receive(socket_num,nlen,LENGTH_SIZE);
  len=0;
  resize_min(nlen);
  len=nlen;

  receive(socket_num,data,len);
  reset_read_head();
}

template<class T>
void octetStream::store(const T& x)
{
    x.pack(*this);
}

template<class T>
T octetStream::get()
{
    T res;
    res.unpack(*this);
    return res;
}

template<class T>
void octetStream::get(T& res)
{
    res.unpack(*this);
}

template<>
inline int octetStream::get()
{
    return get_int(sizeof(int));
}

template<class T>
void octetStream::store(const vector<T>& v)
{
  store(v.size());
  for (auto& x : v)
    store(x);
}

template<class T>
void octetStream::get(vector<T>& v, const T& init)
{
  size_t size;
  get(size);
  v.reserve(size);
  for (size_t i = 0; i < size; i++)
    {
      v.push_back(init);
      get(v.back());
    }
}

template<class T>
void octetStream::get_no_resize(vector<T>& v)
{
  size_t size;
  get(size);
  if (size != v.size())
    throw runtime_error("wrong vector length");
  for (auto& x : v)
    get(x);
}

template<class T, size_t L>
void octetStream::store(const array<T, L>& v)
{
  for (auto& x : v)
    store(x);
}

template<class T, size_t L>
void octetStream::get(array<T, L>& v)
{
  for (auto& x : v)
    get(x);
}

#endif

