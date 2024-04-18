/*
 * NetworkOptions.h
 *
 */

#ifndef TOOLS_NETWORKOPTIONS_H_
#define TOOLS_NETWORKOPTIONS_H_

#include "ezOptionParser.h"
#include "Networking/Server.h"
#include "Networking/Player.h"

#include <string>

class NetworkOptions
{
public:
    int portnum_base;
    std::string hostname;

    NetworkOptions(ez::ezOptionParser& opt, int argc, const char** argv);
};

class NetworkOptionsWithNumber : public NetworkOptions
{
public:
    int nplayers;
    std::string ip_filename;

    NetworkOptionsWithNumber(ez::ezOptionParser& opt, int argc,
            const char** argv, int default_nplayers, bool variable_nplayers);

    Server* start_networking(Names& N, int my_num);
};

#endif /* TOOLS_NETWORKOPTIONS_H_ */
