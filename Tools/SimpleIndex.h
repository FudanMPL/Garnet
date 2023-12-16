// © 2016 Peter Rindal.
// © 2022 Visa.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cryptoTools/Common/Defines.h"
#include "cryptoTools/Common/BitVector.h"
#include "cryptoTools/Common/Matrix.h"

// template <typename T>
typedef uint64_t T;

class SimpleIndex
{
public:
    struct Item
    {
        Item() : mVal(-1) {}

        Item &operator=(const Item &) = default;

        bool isEmpty() const { return mVal == T(-1); }

        // The index is the index of the input that currently
        // occupies this bin position. The index is encode in the
        // first 7 bytes.
        T idx() const { return mVal & (T(-1) >> 8); }

        // The index of the hash function that this item is
        // currently using. This in is encoded in the 8th byte.
        T hashIdx() const { return ((uint8_t *)&mVal)[7] & 127; }

        // Return true if this item was set with a collition.
        bool isCollision() const { return (((uint8_t *)&mVal)[7] >> 7) > 0; }

        // The this item to contain the index idx under the given hash index.
        // The collision value is also encoded.
        void set(T idx, uint8_t hashIdx, bool collision)
        {
            mVal = idx;
            ((uint8_t *)&mVal)[7] = hashIdx | ((collision & 1) << 7);
        }
#ifdef THREAD_SAFE_SIMPLE_INDEX
        Item(const Item &b) : mVal(b.mVal.load(std::memory_order_relaxed)) {}
        Item(Item &&b) : mVal(b.mVal.load(std::memory_order_relaxed)) {}
        std::atomic<T> mVal;
#else
        Item(const Item &b) : mVal(b.mVal) {}
        Item(Item &&b) : mVal(b.mVal) {}
        T mVal;
#endif
    };

    T mMaxBinSize, mNumHashFunctions;

    // The current assignment of items to bins. Only
    // the index of the input item is stored in the bin,
    // not the actual item itself.
    osuCrypto::Matrix<Item> mBins;
    T mNumBins;

    // numBalls x mNumHashFunctions matrix, (i,j) contains the i'th items
    // hash value under hash index j.
    osuCrypto::Matrix<T> mItemToBinMap;

    // The some of each bin.
    std::vector<T> mBinSizes;

    T itemSize;

    osuCrypto::block mHashSeed;
    void print();
    static T get_bin_size(T numBins, T numBalls, T statSecParam, bool approx = true);

    void init(T numBins, T numBalls, T statSecParam = 40, T numHashFunction = 3);
    void insertItems(osuCrypto::span<osuCrypto::block> items, osuCrypto::block hashingSeed);
    void getMapping(std::vector<std::array<T, 3>> &mMapping);
};
