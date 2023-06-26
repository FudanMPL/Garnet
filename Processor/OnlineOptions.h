/*
 * OnlineOptions.h
 *
 */

#ifndef PROCESSOR_ONLINEOPTIONS_H_
#define PROCESSOR_ONLINEOPTIONS_H_

#include "Tools/ezOptionParser.h"
#include "Math/bigint.h"
#include "Math/Setup.h"

class OnlineOptions
{
public:
    static OnlineOptions singleton;

    bool interactive;
    int lgp;
    bigint prime;
    bool live_prep;
    int playerno;
    std::string progname;
    int batch_size;
    std::string memtype;
    bool bits_from_squares;
    bool direct;
    int bucket_size;
    int security_parameter;
    std::string cmd_private_input_file;
    std::string cmd_private_output_file;
    bool verbose;
    bool file_prep_per_thread;
    int trunc_error;
    int opening_sum, max_broadcast;
    bool receive_threads;

    OnlineOptions();
    OnlineOptions(ez::ezOptionParser& opt, int argc, const char** argv,
            bool security);
    OnlineOptions(ez::ezOptionParser& opt, int argc, const char** argv,
            int default_batch_size = 0, bool default_live_prep = true,
            bool variable_prime_length = false, bool security = true);
    template<class T>
    OnlineOptions(ez::ezOptionParser& opt, int argc, const char** argv, T,
            bool default_live_prep = true);
    template<class T>
    OnlineOptions(T);
    ~OnlineOptions() {}

    void finalize(ez::ezOptionParser& opt, int argc, const char** argv);

    void set_trunc_error(ez::ezOptionParser& opt);

    int prime_length();
    int prime_limbs();

    template<class T>
    string prep_dir_prefix(int nplayers)
    {
        int lgp = this->lgp;
        if (prime)
            lgp = numBits(prime);
        return get_prep_sub_dir<T>(PREP_DIR, nplayers, lgp);
    }
};

#endif /* PROCESSOR_ONLINEOPTIONS_H_ */
