#pragma once
#include "cryptoTools/Common/Defines.h"
#include <vector>
#include <cryptoTools/Common/block.h>
#include <cryptoTools/Crypto/PRNG.h>
#include "cryptoTools/Common/BitVector.h"

class PRF {
public:
  PRF(int mseed) {
    low1.resize(128);
    low2.resize(128);
    high1.resize(128);
    high2.resize(128);

    PRNG prng;
    unsigned char seed[16] = {1};
    memcpy(seed, &mseed, sizeof(mseed));
    prng.SetSeed(seed);

    uint64_t tmp;
    for (int i = 0; i < 128; i++) {
      tmp = prng.get<uint64_t>();
      low1[i] = tmp;
      tmp = prng.get<uint64_t>();
      low2[i] = tmp;
      tmp = prng.get<uint64_t>();
      high1[i] = tmp;
      tmp = prng.get<uint64_t>();
      high2[i] = tmp;
    }
  }

  PRF(const std::vector<uint64_t> &tmps_low1,
      const std::vector<uint64_t> &tmps_low2,
      const std::vector<uint64_t> &tmps_high1,
      const std::vector<uint64_t> &tmps_high2)
      : low1(tmps_low1), low2(tmps_low2), high1(tmps_high1), high2(tmps_high2) {
  }

  void compute(uint64_t id, osuCrypto::block &id_prf) const {
    id_prf = osuCrypto::toBlock(0, 0);
    for (int i = 0; i < 64; ++i) {
      bool bit = (id >> i) & 1;
      osuCrypto::block tmp;
      if (bit == 0) {
        tmp = osuCrypto::toBlock(low1[i], high1[i]);
      } else {
        tmp = osuCrypto::toBlock(low2[i], high2[i]);
      }
      id_prf = id_prf ^ tmp;
    }
  }

  void compute(uint64_t id, BitVector &id_prf) const {
    uint64_t rand_x[2] = {0, 0};
    for (int i = 0; i < 64; ++i) {
      bool bit = (id >> i) & 1;
      if (bit == 0) {
        rand_x[0] ^= low1[i];
        rand_x[1] ^= high1[i];
      } else {
        rand_x[0] ^= low2[i];
        rand_x[1] ^= high2[i];
      }
    }
    id_prf = BitVector(reinterpret_cast<uint8_t *>(rand_x), 128);
  }

private:
  std::vector<uint64_t> low1;
  std::vector<uint64_t> low2;
  std::vector<uint64_t> high1;
  std::vector<uint64_t> high2;
};