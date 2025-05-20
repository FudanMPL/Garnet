#ifndef OT_ROW_H_
#define OT_ROW_H_

#include "Math/Z2k.h"
#include "Math/gf2nlong.h"
#define VOLE_HASH_SIZE crypto_generichash_BYTES

template <class T, class U> class DeferredMinus;
template <class T, class U> class DeferredPlus;

template <class T>
class Row
{
public:

    vector<T> rows;

    Row(int size) : rows(size) {}

    Row() : rows() {}

    Row(const vector<T>& _rows) : rows(_rows) {}

    template<class U>
    Row(DeferredMinus<T, U> d) { *this = d; }
    template<class U>
    Row(DeferredPlus<T, U> d) { *this = d; }

    bool operator==(const Row<T>& other) const;
    bool operator!=(const Row<T>& other) const { return not (*this == other); }

    Row<T>& operator+=(const Row<T>& other);
    Row<T>& operator-=(const Row<T>& other);
    
    Row<T>& operator*=(const T& other);

    Row<T> operator*(const T& other);
    DeferredPlus<T, Row<T>> operator+(const Row<T> & other);
    DeferredMinus<T, Row<T>> operator-(const Row<T>& other);

    template<class U>
    Row<T>& operator=(const DeferredMinus<T, U>& d);
    template<class U>
    Row<T>& operator=(const DeferredPlus<T, U>& d);

    Row<T> operator<<(int i) const;

    // fine, since elements in vector are allocated contiguously
    const void* get_ptr() const { return rows[0].get_ptr(); }
    
    void randomize(PRNG& G, size_t size);

    void pack(octetStream& o) const;
    void unpack(octetStream& o);

    size_t size() const { return rows.size(); }

    const T& operator[](size_t i) const { return rows[i]; }

    template <class V>
    friend ostream& operator<<(ostream& o, const Row<V>& x);
};

template <int K>
using Z2kRow = Row<Z2<K>>;

template <class T, class U>
class DeferredMinus
{
public:
    const U& x;
    const Row<T>& y;

    DeferredMinus(const U& x, const Row<T>& y) : x(x), y(y)
    {
        assert(x.size() == y.size());
    }

    size_t size() const
    {
        return x.size();
    }

    T operator[](size_t i) const
    {
        return x[i] - y[i];
    }

    DeferredPlus<T, DeferredMinus> operator+(const Row<T>& other)
    {
        return {*this, other};
    }

    DeferredMinus<T, DeferredMinus> operator-(const Row<T>& other)
    {
        return {*this, other};
    }
};

template <class T, class U>
class DeferredPlus
{
public:
    const U& x;
    const Row<T>& y;

    DeferredPlus(const U& x, const Row<T>& y) : x(x), y(y)
    {
        assert(x.size() == y.size());
    }

    size_t size() const
    {
        return x.size();
    }

    T operator[](size_t i) const
    {
        return x[i] + y[i];
    }

    DeferredPlus<T, DeferredPlus> operator+(const Row<T>& other)
    {
        return {*this, other};
    }

    void pack(octetStream& o) const;
};

#endif /* OT_ROW_H_ */
