#include "cryptoTools/Common/BitVector.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cryptoTools/Common/block.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <openssl/sha.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace std;

// Define a point structure using int64_t for x and BitVector for y
struct Point {
  int64_t x;
  BitVector y;
};

// Random Oracle H
class RandomOracle {
private:
  std::mt19937_64 rng;
  int m; // Output length in bits

public:
  RandomOracle(int m) : m(m) {}

  static uint64_t avalanche64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    // x ^= z;
    // x ^= x >> 33;
    // x *= 0xff51afd7ed558ccdULL;
    // x ^= x >> 33;
    // x *= 0xc4ceb9fe1a85ec53ULL;
    // x ^= x >> 33;
    return z ^ (z >> 31);
  }

  // Hash function implementation (random oracle)
  uint64_t hash(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    unsigned char input[32];
    for (int i = 0; i < 8; i++) {
      input[i] = (a >> (56 - 8 * i)) & 0xff;
      input[i + 8] = (b >> (56 - 8 * i)) & 0xff;
      input[i + 16] = (c >> (56 - 8 * i)) & 0xff;
      input[i + 24] = (d >> (56 - 8 * i)) & 0xff;
    }
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(input, sizeof(input), hash);

    uint64_t result = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
      result = (result * 256 + hash[i]) % m;
    }
    // uint64_t mask = (1ULL << m) - 1;
    return result;
  }

  /**
   * A hash function that maps a pair of uint64_t to a uint64_t.
   *
   * This function is based on the 64-bit hash function from the article
   * "Hash Functions" by Thomas Wang (https://www.concentric.net/~Ttwang/tech/inthash.htm).
   *
   * The function takes two uint64_t values, x and v, and returns a uint64_t value.
   *
   * The hash function first performs a linear combination of x and v using two
   * large prime numbers, then performs a series of bitwise XOR and multiplication
   * operations to increase the dispersion of the hash values.
   *
   * Finally, the function returns the result modulo m, where m is the output length
   * in bits.
   */
  uint64_t hash(uint64_t x, uint64_t v) {
    static const uint64_t MIX1 =
        0x9E3779B185EBCA87ULL; // 大质数, 常被用作哈希混合
    static const uint64_t MIX2 = 0xC2B2AE3D27D4EB4FULL; // 另一个混合常数
    static const uint64_t MIX3 = 0xFF51AFD7ED558CCDLL; // 64 位哈希常见使用
    static const uint64_t MIX4 = 0xC4CEB9FE1A85EC53ULL;

    // 1. 将 X 和 v 先线性混合
    uint64_t h = x * MIX1 + v * MIX2;

    // 2. 进行一系列移位异或和乘法混合，增加分散性
    h ^= (h >> 32);
    h *= MIX3;
    h ^= (h >> 32);
    h *= MIX4;
    h ^= (h >> 32);

    // uint64_t mask = (1ULL << m) - 1;
    return static_cast<uint64_t>(x % m);
  }

  int getOutputLength() const { return m; }
};

class OKVSSender {

public:
  OKVSSender(){};

  static void generateTable(const std::vector<uint64_t> &X,
                            const std::vector<BitVector> &Y, size_t n,
                            std::vector<BitVector> &results, uint64_t &nonce1,
                            uint64_t &nonce2, uint64_t &nonce3) {
    assert(X.size() == Y.size());
    PRNG seed;
    seed.ReSeed();
    // size_t n = static_cast<int>(std::ceil(std::log2(X.size() + 1)));
    RandomOracle H(n);
    // Step : Sample v until all H(F(k, x_i)||v) are distinct
    uint64_t v1, v2, v3;
    std::unordered_map<uint64_t, bool> hash_values;
    bool distinct = false;

    while (!distinct) {
      hash_values.clear();

      // Generate a random v (64-bit)
      v1 = seed.get<uint64_t>();
      v2 = seed.get<uint64_t>();
      v3 = seed.get<uint64_t>();

      // Check if all hash values are distinct
      distinct = true;
      for (const auto &x : X) {
        int64_t hash_val = H.hash(x, v1, v2, v3);

        if (hash_values.find(hash_val) != hash_values.end()) {
          // std::cout << "false, hash_val = " << hash_val << std::endl;
          distinct = false;
          break;
        }

        hash_values[hash_val] = true;
      }
    }
    // std::cout << "finish build okvs\n";
    results.clear();
    // results.resize(n);
    uint64_t rand_x[2];
    BitVector x_bitv;
    // std::cout << "when okvs \n";
    for (size_t i = 0; i < n; i++) {
      // results[i] = seed.get<BitVector>();
      rand_x[0] = seed.get<uint64_t>();
      rand_x[1] = seed.get<uint64_t>();
      x_bitv = BitVector(reinterpret_cast<uint8_t *>(rand_x), 128);
      results.push_back(x_bitv);
      // std::cout << results[i].str() << std::endl;
    }
    for (size_t i = 0; i < X.size(); i++) {
      uint64_t hash_val = H.hash(X[i], v1, v2, v3);
      results[hash_val] = Y[i];
    }
    nonce1 = v1;
    nonce2 = v2;
    nonce3 = v3;
  }
};

class OKVSReceiver {
private:
  std::vector<BitVector> results;
  uint64_t nonce;
  int hashtype = 0;
  uint64_t v1, v2, v3;
  RandomOracle H;

public:
  OKVSReceiver(const std::vector<BitVector> &results, uint64_t nonce)
      : results(results), nonce(nonce), H(results.size()), hashtype(0) {}
  //   BitVector get(int64_t x) {
  //     uint64_t hash_val = H.hash(x ^ nonce);
  //     return results[hash_val];
  //   }
  OKVSReceiver(const std::vector<BitVector> &results, uint64_t v1, uint64_t v2,
               uint64_t v3, uint64_t m)
      : results(results), v1(v1), v2(v2), v3(v3), H(m),
        hashtype(1) {}

  uint64_t getIndex(uint64_t x) {
    // uint64_t hash_val = H.hash(x, nonce);
    // if(hashtype == 0){
    uint64_t hash_val = H.hash(x, v1, v2, v3);
    return hash_val;
  }

  BitVector getByIndex(uint64_t idx) { return results[idx]; }

  std::pair<uint64_t, BitVector> get(uint64_t x) {
    uint64_t hash_val = H.hash(x, v1, v2, v3);
    BitVector y = results[hash_val];
    return {hash_val, y};
  }
};