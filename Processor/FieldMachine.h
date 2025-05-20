/*
 * DishonestMajorityFieldMachine.h
 *
 */

#ifndef PROCESSOR_FIELDMACHINE_H_
#define PROCESSOR_FIELDMACHINE_H_

#include "RingMachine.h"
#include "HonestMajorityMachine.h"
#include "Tools/ezOptionParser.h"
#include "Math/gfp.h"

#include "OnlineOptions.hpp"

template<template<class T> class U, class V = HonestMajorityMachine>
class HonestMajorityFieldMachine
{
public:
    HonestMajorityFieldMachine(int argc, const char** argv);
    HonestMajorityFieldMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            int nplayers = 3);
};

template<template<class T> class U, template<class T> class V, class W, class X = gf2n>
class FieldMachine
{
public:
    FieldMachine(int argc, const char** argv, ez::ezOptionParser& opt,
            OnlineOptions& online_opts, int nplayers = 0);
};

template<template<class U> class T, template<class U> class V = T,
        class W = gf2n, class X = DishonestMajorityMachine>
class DishonestMajorityFieldMachine
{
public:
    DishonestMajorityFieldMachine(int argc, const char** argv,
            ez::ezOptionParser& opt, bool live_prep_default = true)
    {
        OnlineOptions& online_opts = OnlineOptions::singleton;
        online_opts = {opt, argc, argv, T<gfp0>(), live_prep_default};

        FieldMachine<T, V, X, W>(argc, argv, opt, online_opts);
    }
};

#endif /* PROCESSOR_FIELDMACHINE_H_ */
