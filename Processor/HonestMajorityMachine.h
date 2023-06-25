/*
 * DishonestMajorityMachine.h
 *
 */

#ifndef PROCESSOR_HONESTMAJORITYMACHINE_H_
#define PROCESSOR_HONESTMAJORITYMACHINE_H_

#include "OnlineMachine.h"

class HonestMajorityMachine : public OnlineMachine
{
public:
    HonestMajorityMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            OnlineOptions& online_opts, int n_players = 3);

    template<class V>
    HonestMajorityMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            OnlineOptions& online_opts, V, int nplayers) :
            HonestMajorityMachine(argc, argv, opt, online_opts, nplayers)
    {
    }
};

#endif /* PROCESSOR_HONESTMAJORITYMACHINE_H_ */
