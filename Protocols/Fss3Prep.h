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
    // cpu correction word临时
    bigint seed_bigint;
    vector<bool> tcw[2];
    vector<bigint> vcw;
    vector<bigint> scw;
    bigint r_b, r_share, final_cw, u, r_in_2, w, z;
    bool rs_b;
    int dpf_cnt;
    
    Fss3Prep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage), BitPrep<T>(proc, usage),
			RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage), ReplicatedRingPrep<T>(proc, usage)
    {
        this->dpf_cnt = 0;
        int bit_length = T::clear::MAX_N_BITS;
        int input_byte = ceil(bit_length/8.0);
        int batch_size = OnlineOptions::singleton.batch_size;
        if(OnlineOptions::singleton.live_prep){
            this->fss_dpf_eval_values.r_share = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));  
            this->fss_dpf_eval_values.scw = (uint8_t*)malloc(batch_size * bit_length * LAMBDA_BYTE * sizeof(uint8_t));
            this->fss_dpf_eval_values.tcw[0] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            this->fss_dpf_eval_values.tcw[1] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            this->fss_dpf_eval_values.output = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));
            this->fss_dpf_eval_seeds = new aes_eval_block[batch_size];

            this->fss_dpf_gen_values.r = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));  
            this->fss_dpf_gen_values.r_share_0 = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));
            this->fss_dpf_gen_values.r_share_1 = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));
            this->fss_dpf_gen_values.scw = (uint8_t*)malloc(batch_size * bit_length * LAMBDA_BYTE * sizeof(uint8_t));
            this->fss_dpf_gen_values.tcw[0] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            this->fss_dpf_gen_values.tcw[1] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            this->fss_dpf_gen_values.output = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));
            this->fss_dpf_gen_seeds = new aes_gen_block[batch_size];
        }
        else{
            fstream r_in;
            r_in.open("Player-Data/2-fss/r" + to_string(proc->P.my_num()), ios::in);
            r_in >> this->r_share;
            r_in >> this->r_b;
            r_in >> this->rs_b;
            r_in >> this->u;
            r_in >> this->r_in_2;
            r_in >> this->w;
            r_in >> this->z;
            if(proc->P.my_num()!=2){
                bool t0, t1;
                fstream k_in;
                bigint tmp;
                bool tmp_bool;
                // std::cout << "reading k" << proc->P.my_num() << endl;
                k_in.open("Player-Data/2-fss/k" + to_string(proc->P.my_num()), ios::in);
                k_in >> seed_bigint;
                // std::cout << seed_bigint << std::endl;
                for(int i = 1; i < bit_length - 16; i++){
                    k_in >> tmp;
                    this->scw.push_back(tmp);
                    k_in >> tmp;
                    this->vcw.push_back(tmp);
                    k_in >> tmp_bool;
                    this->tcw[0].push_back(tmp_bool);
                    k_in >> tmp_bool;
                    this->tcw[1].push_back(tmp_bool);
                }
                k_in >> final_cw;
                r_in.close();
                k_in.close();
                // std::cout << "finish reading!" << std::endl;
                // std::cout << "preparing fake offline " << batch_size << std::endl;
            }
            // this->fss_dpf_eval_values.r_share = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));  
            // this->fss_dpf_eval_values.scw = (uint8_t*)malloc(batch_size * bit_length * LAMBDA_BYTE * sizeof(uint8_t));
            // this->fss_dpf_eval_values.tcw[0] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            // this->fss_dpf_eval_values.tcw[1] = (bool*)malloc(batch_size * bit_length * sizeof(bool));
            // this->fss_dpf_eval_values.output = (uint8_t*)malloc(batch_size * input_byte * sizeof(uint8_t));
            // this->fss_dpf_eval_seeds = new aes_eval_block[batch_size];
            // bigint seed, r_share, scw, tcw0, tcw1, output;
            // fstream f;
            // string file_name = "";
            // std::cout << batch_size << std::endl;
            // if(proc->P.my_num() == 2)
            //     file_name = "Player-Data/2-fss/dpf_correction_word_" + std::to_string(bit_length) + "_10240_0";
            // else
            //     file_name = "Player-Data/2-fss/dpf_correction_word_" + std::to_string(bit_length) + "_10240_" + std::to_string(proc->P.my_num());
            // std::cout << file_name << std::endl;
            // f.open(file_name, ios::in);
            // for(int i = 0; i < batch_size; i++){
            //     f >> r_share;
            //     bytesFromBigint(&this->fss_dpf_eval_values.r_share[i * input_byte], r_share, input_byte);
            //     f >> seed;
            //     bytesFromBigint(&this->fss_dpf_eval_seeds[i].block[0], seed, LAMBDA_BYTE);
            //     bytesFromBigint(&this->fss_dpf_eval_seeds[i].block[LAMBDA_BYTE], seed, LAMBDA_BYTE);
            //     for(int j = 0; j < bit_length; j++){
            //         f >> scw;
            //         bytesFromBigint(&this->fss_dpf_eval_values.scw[i * bit_length * LAMBDA_BYTE + j * LAMBDA_BYTE], scw, LAMBDA_BYTE);
            //         f >> this->fss_dpf_eval_values.tcw[0][i * bit_length + j];
            //         f >> this->fss_dpf_eval_values.tcw[1][i * bit_length + j];
            //     }
            //     f >> output;
            //     bytesFromBigint(&this->fss_dpf_eval_values.output[i * input_byte], output, input_byte);
            // }
            // f.close();
            // std::cout << "Closed!" << std::endl;
        }
        return;
    }

    void test_insert(int num){
        this->test.push_back(num);
    }

    void buffer_bits() { this->buffer_bits_without_check(); }

    void buffer_dpf_with_gpu(int lambda);

    void gen_fake_dcf(int beta, int lambda);
    
    void gen_dpf_correction_word(Player& P, int gen_num);

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
