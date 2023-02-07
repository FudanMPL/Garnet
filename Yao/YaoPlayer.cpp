/*
 * YaoPlayer.cpp
 *
 */

#include "YaoPlayer.h"
#include "YaoGarbler.h"
#include "YaoEvaluator.h"
#include "Tools/ezOptionParser.h"
#include "Tools/NetworkOptions.h"

#include "GC/Machine.hpp"

YaoPlayer::YaoPlayer(int argc, const char** argv)
{
	ez::ezOptionParser opt;
	opt.add(
			"", // Default.
			0, // Required?
			0, // Number of args expected.
			0, // Delimiter if expecting multiple args.
			"Evaluate only after garbling (default only with multi-threading).", // Help description.
			"-O", // Flag token.
			"--oneshot" // Flag token.
	);
	opt.add(
			"1024", // Default.
			0, // Required?
			1, // Number of args expected.
			0, // Delimiter if expecting multiple args.
			"Minimum number of gates for multithreading (default: 1024).", // Help description.
			"-t", // Flag token.
			"--threshold" // Flag token.
	);
	auto& online_opts = OnlineOptions::singleton;
	online_opts = {opt, argc, argv, false};
	NetworkOptionsWithNumber network_opts(opt, argc, argv, 2, false);
	online_opts.finalize(opt, argc, argv);

	int my_num = online_opts.playerno;
	int threshold;
	bool continuous = not opt.get("-O")->isSet;
	opt.get("-t")->getInt(threshold);
	progname = online_opts.progname;

	GC::ThreadMasterBase* master;
	if (my_num == 0)
	    master = new YaoGarbleMaster(continuous, online_opts, threshold);
	else
	    master = new YaoEvalMaster(continuous, online_opts);

	network_opts.start_networking(master->N, my_num);
	master->run(progname);

	if (my_num == 1)
	    ((YaoEvalMaster*)master)->machine.write_memory(0);

	delete master;
}

YaoPlayer::~YaoPlayer()
{
}
