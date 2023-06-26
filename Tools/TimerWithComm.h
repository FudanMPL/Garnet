/*
 * TimerWithComm.h
 *
 */

#ifndef TOOLS_TIMERWITHCOMM_H_
#define TOOLS_TIMERWITHCOMM_H_

#include "time-func.h"
#include "Networking/Player.h"

class TimerWithComm : public Timer
{
    NamedCommStats total_stats, last_stats;

public:
    void start(const NamedCommStats& stats = {});
    void stop(const NamedCommStats& stats = {});

    double mb_sent();
};

#endif /* TOOLS_TIMERWITHCOMM_H_ */
