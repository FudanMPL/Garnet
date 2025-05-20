/*
 * Exceptions.cpp
 *
 */

#include "Exceptions.h"
#include "Math/bigint.h"

IO_Error::IO_Error(const string& m)
{
    ans = "IO-Error : " + m;
}

file_error::file_error(const string& m)
{
    ans = "File Error : " + m;
}

Processor_Error::Processor_Error(const string& m)
{
    msg = "Processor-Error : " + m;
}

Processor_Error::Processor_Error(const char* m) :
        Processor_Error(string(m))
{
}

wrong_gfp_size::wrong_gfp_size(const char* name, const bigint& p,
        const char* symbol, int n_limbs) :
        runtime_error(
                string() + name + " wrong size for modulus " + to_string(p)
                        + ". Maybe change " + symbol + " to "
                        + to_string(n_limbs) + ".")
{
}

overflow::overflow(const string& name, size_t i, size_t n) :
        runtime_error(name + " overflow: " + to_string(i) + "/" + to_string(n))
{
}

unknown_input_type::unknown_input_type(int type) :
        runtime_error("unkown type: " + to_string(type))
{
}

invalid_opcode::invalid_opcode(int opcode) :
        runtime_error("invalid opcode: " + to_string(opcode))
{
}

input_error::input_error(const char* name, const string& filename,
        istream& input_file, size_t input_counter)
{
    input_file.clear();
    string token;
    input_file >> token;
    msg += string() + "cannot read " + name + " from " + filename
            + ", problem with '" + token + "' after "
            + to_string(input_counter);
}

signature_mismatch::signature_mismatch(const string& filename) :
        runtime_error("Signature in " + filename + " doesn't match protocol. "
                "Re-run preprocessing")
{
}

insufficient_memory::insufficient_memory(size_t size, const string& type) :
        runtime_error(
                "program requires too much " + type + " memory: "
                        + to_string(size))
{
}

not_enough_to_buffer::not_enough_to_buffer(const string& type, const string& filename)  :
        runtime_error(
                "Not enough data available for buffer"
                        + (filename.empty() ? "" : (" in " + filename)) + ". "
                                "Maybe insufficient preprocessing" + type
                        + ".\nFor benchmarking, you can activate reusing data by "
                                "adding -DINSECURE to the compiler options.")
{
}

gf2n_not_supported::gf2n_not_supported(int n, string options) :
        runtime_error(
                "GF(2^" + to_string(n) + ") not supported"
                        + (options.empty() ? "" : ", options are " + options))
{
}
