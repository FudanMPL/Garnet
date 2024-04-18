/*
 * Buffer.cpp
 *
 */

#include "Tools/Buffer.h"
#include "Processor/BaseMachine.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

bool BufferBase::rewind = false;


void BufferBase::setup(ifstream* f, int length, const string& filename,
        const char* type, const string& field)
{
    file = f;
    tuple_length = length;
    data_type = type;
    field_type = field;
    this->filename = filename;
}

bool BufferBase::is_pipe()
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) == 0)
        return S_ISFIFO(buf.st_mode);
    else
        return false;
}

void BufferBase::seekg(int pos)
{
    assert(not is_pipe());

#ifdef DEBUG_BUFFER
    if (pos != 0)
        printf("seek %d %s thread %d\n", pos, filename.c_str(),
                BaseMachine::thread_num);
#endif
    if (not file)
    {
        if (pos == 0)
            return;
        else
            file = open();
    }

    file->seekg(header_length + pos * tuple_length);
    if (file->eof() || file->fail())
    {
        // let it go in case we don't need it anyway
        if (pos != 0)
            try_rewind();
    }
#ifdef DEBUG_BUFFER
    printf("seek %d %d thread %d\n", pos, int(file->tellg()),
            BaseMachine::thread_num);
#endif
    next = BUFFER_SIZE;
}

void BufferBase::try_rewind()
{
    assert(not is_pipe());

#ifndef INSECURE
    string type;
    if (field_type.size() and data_type.size())
        type = (string)" of " + field_type + " " + data_type;
    throw not_enough_to_buffer(type, filename);
#endif
    file->clear(); // unset EOF flag
    file->seekg(header_length);
    if (file->peek() == ifstream::traits_type::eof())
        throw runtime_error("empty file: " + filename);
    if (!rewind)
        cerr << "REWINDING - ONLY FOR BENCHMARKING" << endl;
    rewind = true;
    eof = true;
}

void BufferBase::prune()
{
    // only prune in secure mode
#ifdef INSECURE
    return;
#endif

    if (is_pipe())
        return;

    if (file and (not file->good() or file->peek() == EOF))
        purge();
    else if (file and file->tellg() != header_length)
    {
#ifdef VERBOSE
        cerr << "Pruning " << filename << endl;
#endif
        string tmp_name = filename + ".new";
        ofstream tmp(tmp_name.c_str());
        size_t start = file->tellg();
        char buf[header_length];
        file->seekg(0);
        file->read(buf, header_length);
        tmp.write(buf, header_length);
        file->seekg(start);
        tmp << file->rdbuf();
        if (tmp.fail())
            throw runtime_error(
                    "problem writing to " + tmp_name + " from "
                            + to_string(start) + " of " + filename);
        tmp.close();
        file->close();
        rename(tmp_name.c_str(), filename.c_str());
        file->open(filename.c_str(), ios::in | ios::binary);
    }
#ifdef VERBOSE
    else
    {
        cerr << "Not pruning " << filename << " because it's ";
        if (file)
            cerr << "closed";
        else
            cerr << "unused";
        cerr << endl;
    }
#endif
}

void BufferBase::purge()
{
    if (file and not is_pipe())
    {
#ifdef VERBOSE
        cerr << "Removing " << filename << endl;
#endif
        unlink(filename.c_str());
        file->close();
        file = 0;
    }
}

void BufferBase::check_tuple_length(int tuple_length)
{
    if (tuple_length != this->tuple_length)
        throw Processor_Error("inconsistent tuple length");
}
