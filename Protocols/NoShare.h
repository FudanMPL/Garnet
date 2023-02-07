/*
 * NoShare.h
 *
 */

#ifndef PROTOCOLS_NOSHARE_H_
#define PROTOCOLS_NOSHARE_H_

#include "ShareInterface.h"
#include "Math/bigint.h"
#include "Math/gfp.h"
#include "GC/NoShare.h"
#include "BMR/Register.h"

#include "NoLivePrep.h"
#include "NoProtocol.h"

template<class T>
class NoShare : public ShareInterface
{
    typedef NoShare This;

public:
    // type for clear values in relevant domain
    typedef T clear;
    typedef clear open_type;

    // disable binary computation
    typedef GC::NoShare bit_type;

    // opening facility
    typedef NoOutput<NoShare> MAC_Check;
    typedef MAC_Check Direct_MC;

    // multiplication protocol
    typedef NoProtocol<NoShare> Protocol;

    // preprocessing facility
    typedef NoLivePrep<NoShare> LivePrep;

    // private input facility
    typedef NoInput<NoShare> Input;

    // default private output facility (using input tuples)
    typedef ::PrivateOutput<NoShare> PrivateOutput;

    // description used for debugging output
    static string type_string()
    {
        return "no share";
    }

    // used for preprocessing storage location
    static string type_short()
    {
        return "no";
    }

    // size in bytes
    // must match assign/pack/unpack and machine-readable input/output
    static int size()
    {
        // works only if purely on the stack
        // skip one byte for ShareInterface
        return sizeof(This) - 1;
    }

    // maximum number of corrupted parties
    // only used in virtual machine instruction
    static int threshold(int)
    {
        throw runtime_error("no threshold");
        return -1;
    }

    // serialize computation domain for client communication
    static void specification(octetStream& os)
    {
        T::specification(os);
    }

    // constant secret sharing
    static This constant(const clear&, int, const mac_key_type&)
    {
        throw runtime_error("no constant sharing");
        return {};
    }

    // share addition
    This operator+(const This&)
    {
        throw runtime_error("no share addition");
        return {};
    }

    // share subtraction
    This operator-(const This&)
    {
        throw runtime_error("no share subtraction");
        return {};
    }

    This& operator+=(const This& other)
    {
        *this = *this + other;
        return *this;
    }

    This& operator-=(const This& other)
    {
        *this = *this - other;
        return *this;
    }

    // private-public multiplication
    This operator*(const clear&) const
    {
        throw runtime_error("no private-public multiplication");
        return {};
    }

    // private-public division
    This operator/(const clear&) const
    {
        throw runtime_error("no private-public division");
        return {};
    }

    // multiplication by power of two
    This operator<<(int) const
    {
        throw runtime_error("no right shift");
        return {};
    }

    // assignment from byte string
    // must match unpack
    void assign(const char* buffer)
    {
        // works only if purely on the stack
        // skip one byte for ShareInterface
        memcpy((char*) this + 1, buffer, size());
    }

    // serialization
    // must use the number of bytes given by size()
    void pack(octetStream& os, bool = false) const
    {
        // works only if purely on the stack
        // skip one byte for ShareInterface
        memcpy(os.append(size()), (char*) this + 1, size());
    }

    // serialization
    // must use the number of bytes given by size()
    void unpack(octetStream& os, bool = false)
    {
        assign((char*)os.consume(size()));
    }

    // serialization
    // must use the number of bytes given by size() for human=false
    void input(istream& is, bool human)
    {
        if (human)
            throw runtime_error("no human-readable input");
        else
        {
            char buf[size()];
            is.read(buf, size());
            assign(buf);
        }
    }

    // serialization
    // must use the number of bytes given by size() for human=false
    void output(ostream& os, bool human) const
    {
        if (human)
            throw runtime_error("no human-readable output");
        else
        {
            octetStream buf;
            pack(buf);
            os.write((char*)buf.get_data(), size());
        }
    }
};

#endif /* PROTOCOLS_NOSHARE_H_ */
