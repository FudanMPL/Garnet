/*
 * FlexBuffer.cpp
 *
 */

#include "FlexBuffer.h"
#include <iostream>
#include <unistd.h>
#include "BMR/network/utils.h"
using namespace std;

#ifndef BUFFER_DIR
#define BUFFER_DIR "/tmp"
#endif

ReceivedMsgStore::~ReceivedMsgStore()
{
#ifdef VERBOSE
	cerr << "Stored " << (double)total_size / 1e9 << " GB in "
			<< push_timer.elapsed() << " seconds and retrieved them in "
			<< pop_timer.elapsed() << " seconds " << endl;
#endif
}

void ReceivedMsgStore::push(ReceivedMsg& msg)
{
#ifdef DEBUG_STORE
    cout << "pushing msg of length " << msg.size() << endl;
    //phex(msg.data(), min(100UL, msg.size()));
#endif
    TimeScope ts(push_timer);
	total_size += msg.size();
	if (mem_size != N and files.empty())
	{
		mem[(start + mem_size) % N] = msg;
		mem_size++;
	}
	else
	{
		char filename[1000];
		sprintf(filename, "%s/%d.XXXXXX", BUFFER_DIR, getpid());
		FILE* file = fdopen(mkstemp(filename), "w");
		if (!file)
			throw runtime_error("can't open file");
		size_t len = msg.size();
		size_t ptr = msg.ptr - msg.buf;
		if (fwrite(&len, sizeof(len), 1, file) != 1)
			throw runtime_error("can't write");
		if (fwrite(&ptr, sizeof(len), 1, file) != 1)
			throw runtime_error("can't write");
		if (len != 0)
		    if (fwrite(msg.data(), msg.size(), 1, file) != 1)
		        throw runtime_error("can't write");
		if (fclose(file) != 0)
			throw runtime_error("can't close");
		files.push_back(filename);
	}
}

bool ReceivedMsgStore::pop(ReceivedMsg& msg)
{
	TimeScope ts(pop_timer);
	if (mem_size != 0)
	{
		msg = mem[start];
		start = (start + 1) % N;
		mem_size--;
#ifdef DEBUG_STORE
	    cout << "popping from memory msg of length " << msg.size() << endl;
	    //phex(msg.data(), min(100UL, msg.size()));
#endif
		return true;
	}
	else if (!files.empty())
	{
		string filename = files.front();
		FILE* file = fopen(filename.c_str(), "r");
		files.pop_front();
		fseek(file, 0, SEEK_SET);
		size_t len;
		if (fread(&len, sizeof(len), 1, file) != 1)
		{
			perror("can't read length");
			throw runtime_error("can't read length");
		}
		msg.resize(len);
		size_t ptr;
		if (fread(&ptr, sizeof(len), 1, file) != 1)
		{
			perror("can't read length");
			throw runtime_error("can't read length");
		}
		msg.ptr = msg.buf + ptr;
		if (len != 0)
		    if (fread(msg.data(), len, 1, file) != 1)
		    {
		        perror("can't read data");
		        throw runtime_error("can't read data");
		    }
		fclose(file);
		remove(filename.c_str());
#ifdef DEBUG_FLEXBUF
        cout << "popping from disk msg of length " << msg.size() << endl;
        phex(msg.data(), min(100UL, msg.size()));
#endif
		return true;
	}
	return false;
}
