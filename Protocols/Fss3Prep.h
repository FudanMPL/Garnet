/*
 * Fss3Prep.h
 *
 */

#ifndef PROTOCOLS_FSS3PREP_H_
#define PROTOCOLS_FSS3PREP_H_

#include "Processor/OnlineOptions.h"
#include "ReplicatedPrep.h"
#include "GC/SemiSecret.h"
#include "Tools/aes.h"
#include <vector>
#include <utility>
#include <iostream>
#include "GPU/dynamic_dpf/fss_struct.h"
#include <typeinfo>
#include <fstream>

template<class T>
class Fss3Prep : public virtual SemiHonestRingPrep<T>,
        public virtual ReplicatedRingPrep<T>
{
    
    void buffer_dabits(ThreadQueues*);

public:
    aes_gen_block * fss_dpf_gen_seeds;
    aes_eval_block * fss_dpf_eval_seeds;
    InputByteRelatedValuesGen fss_dpf_gen_values;
    InputByteRelatedValuesEval fss_dpf_eval_values;
    
    // Sigma 去除最高位和小数位
    bigint seed_bigint;
    vector<bool> tcw[2];
    vector<bigint> vcw;
    vector<bigint> scw;
    // Sigma 去除最高位
    bigint full_seed_bigint;
    vector<bool> full_tcw[2];
    vector<bigint> full_vcw;
    vector<bigint> full_scw;
    // dcf values
    bigint r_b, r_share, final_cw, u, r_in_2, w, z;
    // conv_relu values
    bigint r_select_share, r_mask_share, r_drelu_share, u_select_share, o_select_share, p_select_share, v_select_share, w_select_share, reverse_u_select_share, reverse_1_u_select_share, full_final_cw, reshare_value[2];
    bool rs_b;
    int dpf_cnt;
    
    Fss3Prep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage), BitPrep<T>(proc, usage),
			RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage), ReplicatedRingPrep<T>(proc, usage)
    {
        if(OnlineOptions::singleton.live_prep){
            std::cout << "unsupport online now" << std::endl; 
        }
        return;    
    }

    void test_insert(int num){
        this->test.push_back(num);
    }

    void buffer_bits() { this->buffer_bits_without_check(); }

    void buffer_dpf_with_gpu(int lambda);

    void gen_fake_dcf(int beta, int n);
    
    void gen_fake_conv_relu(int beta, int n, int float_bits);

    void gen_fake_trunc(int float_bits);

    void init_offline_values(SubProcessor<T>* proc, int init_case);

    // void gen_dpf_correction_word(Player& P, int gen_num);

    void get_one_no_count(Dtype dtype, T& a)
    {
        if (dtype ==  DATA_BIT){
            typename T::bit_type b;
            this->get_dabit_no_count(a, b);
        }
        else{
            throw not_implemented();
        }
    }

    void get_correction_word_no_count(Dtype dtype);
};

#endif /* PROTOCOLS_FSS3PREP_H_ */
