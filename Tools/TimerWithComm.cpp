/*
 * TimerWithComm.cpp
 *
 */

#include "TimerWithComm.h"

void TimerWithComm::start(const NamedCommStats& stats)
{
    Timer::start();
    last_stats = stats;
}

void TimerWithComm::stop(const NamedCommStats& stats)
{
    Timer::stop();
    total_stats += stats - last_stats;
}

double TimerWithComm::mb_sent()
{
    return total_stats.sent * 1e-6;
}
