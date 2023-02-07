/*
 * SubDataFilesBase.cpp
 *
 */

#include "PrepBase.h"

#include "Data_Files.h"
#include "OnlineOptions.h"

string PrepBase::get_suffix(int thread_num)
{
    if (OnlineOptions::singleton.file_prep_per_thread)
    {
        assert(thread_num >= 0);
        return "-T" + to_string(thread_num);
    }
    else
        return "";
}

string PrepBase::get_filename(const string& prep_data_dir,
        Dtype dtype, const string& type_short, int my_num, int thread_num)
{
    return prep_data_dir + DataPositions::dtype_names[dtype] + "-" + type_short
            + "-P" + to_string(my_num) + get_suffix(thread_num);
}

string PrepBase::get_input_filename(const string& prep_data_dir,
        const string& type_short, int input_player, int my_num, int thread_num)
{
    return prep_data_dir + "Inputs-" + type_short + "-P" + to_string(my_num)
            + "-" + to_string(input_player) + get_suffix(thread_num);
}

string PrepBase::get_edabit_filename(const string& prep_data_dir,
        int n_bits, int my_num, int thread_num)
{
    return prep_data_dir + "edaBits-" + to_string(n_bits) + "-P"
            + to_string(my_num) + get_suffix(thread_num);
}

void PrepBase::print_left(const char* name, size_t n, const string& type_string,
        size_t used)
{
    if (n > 0 and OnlineOptions::singleton.verbose)
        cerr << "\t" << n << " " << name << " of " << type_string << " left"
                << endl;

    if (n > used / 10)
        cerr << "Significant amount of unused " << name << " of " << type_string
                << ". For more accurate benchmarks, "
                << "consider reducing the batch size with -b." << endl;
}

void PrepBase::print_left_edabits(size_t n, size_t n_batch, bool strict,
        int n_bits, size_t used)
{
    if (n > 0 and OnlineOptions::singleton.verbose)
    {
        cerr << "\t~" << n * n_batch;
        if (not strict)
            cerr << " loose";
        cerr << " edaBits of size " << n_bits << " left" << endl;
    }

    if (n > used / 10)
        cerr << "Significant amount of unused edaBits of size " << n_bits
                << ". For more accurate benchmarks, "
                << "consider reducing the batch size with -b "
                << "or increasing the bucket size with -B." << endl;
}
