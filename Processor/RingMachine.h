/*
 * ReplicatedMachine.h
 *
 */

#ifndef PROCESSOR_RINGMACHINE_H_
#define PROCESSOR_RINGMACHINE_H_

#include <string>
using namespace std;

#include "Tools/ezOptionParser.h"
#include "Processor/OnlineOptions.h"
#include "Networking/Player.h"
#include "OnlineMachine.h"

template<template<int L> class U, template<class T> class V>
class HonestMajorityRingMachine
{
public:
    HonestMajorityRingMachine(int argc, const char** argv, int nplayers = 3);
    HonestMajorityRingMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            int nplayers = 3);
};

template<template<int L> class U, template<class T> class V, class W>
class RingMachine
{
public:
    RingMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            OnlineOptions& online_opts, int nplayers = 0);
};

template<template<int K, int S> class U, template<class T> class V>
class HonestMajorityRingMachineWithSecurity
{
public:
    HonestMajorityRingMachineWithSecurity(int argc, const char** argv,
            ez::ezOptionParser& opt);
};

template<template<int K> class T, template<class V> class U>
class DishonestMajorityRingMachine
{
public:
    DishonestMajorityRingMachine(int argc, const char** argv,
            ez::ezOptionParser& opt, bool live_prep_default = true)
    {
        OnlineOptions& online_opts = OnlineOptions::singleton;
        online_opts = {opt, argc, argv, T<64>(), live_prep_default};

        RingMachine<T, U, DishonestMajorityMachine>(argc, argv, opt, online_opts);
    }
};

#endif /* PROCESSOR_RINGMACHINE_H_ */
