/*
 * Vss-Field-Party.h
 *
 */

#ifndef MACHINES_VSSFIELDPARTY_H_
#define MACHINES_VSSFIELDPARTY_H_

#include "Tools/ezOptionParser.h"

class VssFieldOptions
{
public:
    static VssFieldOptions singleton;
    static VssFieldOptions& s();

    int nparties;
    int npparties;   // the number of privileged parties
    int naparties;   // the number of assistant parties 
    int ndparties;   // the number of assistant parties allowed to drop out

    VssFieldOptions(int nparties = 3, int npparties = 1, int naparties = 2, int ndparties = 1);
    VssFieldOptions(ez::ezOptionParser& opt, int argc, const char** argv);

    void set_naparties(ez::ezOptionParser& opt);
    void set_ndparties(ez::ezOptionParser& opt);
    void set_npparties(ez::ezOptionParser& opt);
};

class VssFieldMachine : public VssFieldOptions
{
};

template<template<class U> class T>
class VssFieldMachineSpec
{
public:
    VssFieldMachineSpec(int argc, const char** argv);
};

#endif /* MACHINES_VSSFIELDPARTY_H_ */
