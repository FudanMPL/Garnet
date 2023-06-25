/*
 * MascotParams.h
 *
 */

#ifndef OT_MASCOTPARAMS_H_
#define OT_MASCOTPARAMS_H_

#include "Tools/OfflineMachineBase.h"

class MascotParams : virtual public OfflineParams
{
public:
    string prep_data_dir;
    bool generateMACs;
    bool amplify;
    bool check;
    bool correlation_check;
    bool generateBits;
    bool use_extension;
    bool fewer_rounds;
    bool fiat_shamir;
    struct timeval start, stop;

    MascotParams();

    void set_passive();
};

#endif /* OT_MASCOTPARAMS_H_ */
