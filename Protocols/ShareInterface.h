/*
 * ShareInterface.h
 *
 */

#ifndef PROTOCOLS_SHAREINTERFACE_H_
#define PROTOCOLS_SHAREINTERFACE_H_

#include <vector>
#include <string>
#include <stdexcept>
using namespace std;

#include "Tools/Exceptions.h"

class Player;
class Instruction;
class ValueInterface;

namespace GC
{
class NoShare;
class NoValue;
}

class ShareInterface
{
public:
    typedef GC::NoShare part_type;
    typedef GC::NoShare bit_type;

    typedef GC::NoValue mac_key_type;
    typedef GC::NoShare mac_type;
    typedef GC::NoShare mac_share_type;

    static const bool needs_ot = false;
    static const bool expensive = false;
    static const bool expensive_triples = false;

    static const bool has_trunc_pr = false;
    static const bool has_split = false;
    static const bool has_mac = false;
    static const bool malicious = false;

    static const false_type triple_matmul;

    const static bool symmetric = true;

    static const int default_length = 1;

    static string type_short() { throw runtime_error("shorthand undefined"); }

    static bool real_shares(const Player&) { return true; }

    template<class T, class U>
    static void split(vector<U>, vector<int>, int, T*, int,
            typename U::Protocol&)
    { throw runtime_error("split not implemented"); }

    template<class T>
    static void shrsi(T&, const Instruction&)
    { throw runtime_error("shrsi not implemented"); }

    static bool get_rec_factor(int, int) { return false; }

    template<class T>
    static void read_or_generate_mac_key(const string&, const Player&, T&) {}

    template<class T, class U>
    static void generate_mac_key(T&, U&) {}

    static int threshold(int) { throw runtime_error("undefined threshold"); }
};

#endif /* PROTOCOLS_SHAREINTERFACE_H_ */
