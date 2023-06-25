/*
 * VssMC.h
 *
 */

#ifndef PROTOCOLS_VSSMC_H_
#define PROTOCOLS_VSSMC_H_

#include "MAC_Check.h"
#include "Tools/Bundle.h"

/**
 * Vector space secret sharing opening protocol (indirect communication)
 */
template<class T>
class VssMC : public TreeVss_Sum<typename T::open_type>, public MAC_Check_Base<T>
{
protected:
    vector<int> lengths;

public:
    // emulate MAC_Check
    VssMC(const typename T::mac_key_type& _ = {}, int __ = 0, int ___ = 0)
    { (void)_; (void)__; (void)___; }
    virtual ~VssMC() {}

    virtual void init_open(const Player& P, int n = 0);
    virtual void prepare_open(const T& secret, int n_bits = -1);
    virtual void exchange(const Player& P);

    void Check(const Player& P) { (void)P; }

    VssMC& get_part_MC() { return *this; }
};

/**
 * Vector space secret sharing opening protocol (direct communication)
 */

// to do 
template<class T>
class DirectVssMC : public VssMC<T>
{
public:
    DirectVssMC() {}
    // emulate Direct_MAC_Check
    DirectVssMC(const typename T::mac_key_type&, const Names& = {}, int = 0, int = 0) {}

    void POpen_(vector<typename T::open_type>& values,const vector<T>& S,const PlayerBase& P);
    void POpen(vector<typename T::open_type>& values,const vector<T>& S,const Player& P)
    { POpen_(values, S, P); }
    void POpen_Begin(vector<typename T::open_type>& values,const vector<T>& S,const Player& P);
    void POpen_End(vector<typename T::open_type>& values,const vector<T>& S,const Player& P);

    void exchange(const Player& P) { exchange_(P); }
    void exchange_(const PlayerBase& P);

    void Check(const Player& P) { (void)P; }
};

#endif /* PROTOCOLS_VSSMC_H_ */
