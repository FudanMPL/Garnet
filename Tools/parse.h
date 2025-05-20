/*
 * parse.h
 *
 */

#ifndef TOOLS_PARSE_H_
#define TOOLS_PARSE_H_

#include <iostream>
#include <vector>
using namespace std;

#ifdef __APPLE__
# include <libkern/OSByteOrder.h>
#define be32toh(x) OSSwapBigToHostInt32(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#endif

// Read a 4-byte integer
inline int get_int(istream& s)
{
  int n;
  s.read((char*) &n, 4);
  return be32toh(n);
}

// Read an 8-byte integer
inline int64_t get_long(istream& s)
{
  int64_t n;
  s.read((char*) &n, 8);
  return be64toh(n);
}

// Read several integers
inline void get_ints(int* res, istream& s, int count)
{
  s.read((char*) res, 4 * count);
  for (int i = 0; i < count; i++)
    res[i] = be32toh(res[i]);
}

inline void get_vector(int m, vector<int>& start, istream& s)
{
  if (s.fail())
    throw runtime_error("error when parsing vector");
  start.resize(m);
  s.read((char*) start.data(), 4 * m);
  for (int i = 0; i < m; i++)
    start[i] = be32toh(start[i]);
}

#endif /* TOOLS_PARSE_H_ */
