/*
 * Element.h
 *
 */

#ifndef ECDSA_P256ELEMENT_H_
#define ECDSA_P256ELEMENT_H_

#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#include "Math/gfp.h"

class P256Element : public ValueInterface
{
public:
    typedef gfp_<2, 4> Scalar;

private:
    static EC_GROUP* curve;

    EC_POINT* point;

public:
    typedef P256Element next;
    typedef void Square;

    static const true_type invertible;

    static int size() { return 0; }
    static int length() { return 256; }
    static string type_string() { return "P256"; }

    static void init();
    static void finish();

    P256Element();
    P256Element(const P256Element& other);
    P256Element(const Scalar& other);
    P256Element(word other);
    ~P256Element();

    P256Element& operator=(const P256Element& other);

    void check();

    Scalar x() const;

    P256Element operator+(const P256Element& other) const;
    P256Element operator-(const P256Element& other) const;
    P256Element operator*(const Scalar& other) const;

    P256Element& operator+=(const P256Element& other);
    P256Element& operator/=(const Scalar& other);

    bool operator==(const P256Element& other) const;
    bool operator!=(const P256Element& other) const;

    void assign_zero() { *this = {}; }
    bool is_zero() { return *this == P256Element(); }
    void add(octetStream& os) { *this += os.get<P256Element>(); }

    void pack(octetStream& os, int = -1) const;
    void unpack(octetStream& os, int = -1);

    octetStream hash(size_t n_bytes) const;

    friend ostream& operator<<(ostream& s, const P256Element& x);
};

P256Element operator*(const P256Element::Scalar& x, const P256Element& y);

#endif /* ECDSA_P256ELEMENT_H_ */
