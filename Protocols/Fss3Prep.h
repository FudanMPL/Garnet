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
    bigint r_b, r_share, final_cw, u, r_in_2, w, z, rr_share, rr_share_tmp;
    // conv_relu values
    bigint r_select_share, r_mask_share, r_drelu_share, u_select_share, o_select_share, p_select_share, v_select_share, w_select_share, reverse_u_select_share, reverse_1_u_select_share, full_final_cw, reshare_value, reshare_value_tmp;
    bool rs_b;
    int dpf_cnt;
    
    Fss3Prep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage), BitPrep<T>(proc, usage),
			RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage), ReplicatedRingPrep<T>(proc, usage)
    {
        if(~OnlineOptions::singleton.live_prep){
            this->full_scw = {bigint("19576590795982552526918330347417303758"),bigint("152005102853457672922634509585678210343"),bigint("133030167457342557950692980351772296025"),bigint("169104242315685039096137806069153614778"),bigint("113272130752711237789286901523706460418"),bigint("166423914147056779539840487365890136503"),bigint("97764632108818560764464102285068500044"),bigint("45170401086954320413318697667821645708"),bigint("94323384931775993944742670591319454047"),bigint("23441449537169823249119315533230424421"),bigint("82060963419604823294375345024464506362"),bigint("11350006288213106579930595105685305571"),bigint("69801969085942469795957692274392903773"),bigint("13923011497514292125276585690007357000"),bigint("27577777490293321209093213583358103751"),bigint("16845597454480539747828252009395498598"),bigint("83658298154652757064546194760794752030"),bigint("105146982710934336134889260079704692097"),bigint("82364022520629363275978382533681491901"),bigint("147410199745147005860382935705233204709"),bigint("18073290882410316869752577809425522197"),bigint("55257461098480952353783550178787110050"),bigint("90373262861912551512176669739344562700"),bigint("125986156062492574881230776603129555317"),bigint("27387993362999577341545881984317692194"),bigint("159236285653673888237559076417773554511"),bigint("139810766839008442091515069572148519318"),bigint("133111572765744711047685987360959463062"),bigint("9145995575716464925056762452115434482"),bigint("133826683280710783255244156949458238074"),bigint("47347517047515398140086323103330444398"),bigint("37751354021320342973327857838814885295"),bigint("49743726597945435727977607332802525518"),bigint("157007366182293249367272969581332395303"),bigint("153316721447728499003349281191129701938"),bigint("84307069445608947848080101119651876716"),bigint("99326225723651284298652252376981386309"),bigint("108647007620487102810435899843032590182"),bigint("8887677146825653804162025648985118911"),bigint("31482007124699008193738011472905213057"),bigint("96206841444840544130957578423057431107"),bigint("92183083581110707650498715542515287802"),bigint("122699749848955623832366006781756973272"),bigint("70506179001438421742652931270876270180"),bigint("15005635016202590428998713141577974346"),bigint("12024517123405994331788974187957607408"),bigint("118983253869378795347870966907085804068"),bigint("148277379103809753330809342280939222705"),bigint("131031758564502644104344422592136623718"),bigint("52469145531641176096728952352315203077"),bigint("14769182541612248864566713273931692783"),bigint("62163954008804686603734817615876768240"),bigint("126797909416683686890863391359036581940"),bigint("121925757772429564033627327876841490222"),bigint("102562376501282000360071431586446212567"),bigint("36751382113074267870210230840395553002"),bigint("43175176388416645551924907584892873780"),bigint("128943965284363419004678797096026147221"),bigint("17755598437038057175425405863388056433"),bigint("90714610769705452919598111251666393298"),bigint("69880320665606104365018270803260353890"),bigint("13928625489141409690023114563328524059")};
            if(proc->P.my_num()==0){
                this->full_seed_bigint = bigint("85905340171514517502442653763461977650");
                this->full_vcw = {0, 2, 1, 1, -2, -1, 0, -1, 1, -3, -3, 1, 2, 1, -1, 0, 0, 1, 0, -2, 0, 1, 1, -1, 2, 2, 1, -1, -1, -1, 1, 0, -3, 0, -1, 2, 0, -1, -2, 1, 2, 0, 0, -3, -1, 1, 1, 1, 1, 2, -2, 0, 3, 4, -3, -2, -2, -2, 1, 1, 1, 0};
                this->full_tcw[0] = {0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1};
                this->full_tcw[1] = {1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1};
                this->full_final_cw = 1;
                this->reshare_value = bigint("17291198612630036802");
                this->r_mask_share = bigint("10109688523258868159");
                this->r_drelu_share = 0;
                this->r_select_share = 1;
                this->u_select_share = bigint("-10590238671395853096");
                this->reverse_u_select_share = bigint("-7688260422197370906");
                this->reverse_1_u_select_share = bigint("-10911337949929667508");
                this->o_select_share = bigint("-5652449027981494062");
                this->p_select_share = bigint("-14522637960414365351");
                this->v_select_share = bigint("-4689535693517965557");
                this->w_select_share = bigint("-7875981768081349498");
            }
            else if(proc->P.my_num()==1){
                this->full_seed_bigint = bigint("7446298975748891766299871760332609212");
                this->full_vcw = {0, 2, 1, 1, -2, -1, 0, -1, 1, -3, -3, 1, 2, 1, -1, 0, 0, 1, 0, -2, 0, 1, 1, -1, 2, 2, 1, -1, -1, -1, 1, 0, -3, 0, -1, 2, 0, -1, -2, 1, 2, 0, 0, -3, -1, 1, 1, 1, 1, 2, -2, 0, 3, 4, -3, -2, -2, -2, 1, 1, 1, 0};
                this->full_tcw[0] = {0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1};
                this->full_tcw[1] = {1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1};
                this->full_final_cw = 1;
                this->reshare_value = bigint("5193242797907115985");
                this->r_mask_share = bigint("3694772080561006902");
                this->r_drelu_share = 1;
                this->r_select_share = 0;
                this->u_select_share = bigint("10590238671395853097");
                this->reverse_u_select_share = bigint("7688260422197370906");
                this->reverse_1_u_select_share = bigint("10911337949929667508");
                this->o_select_share = bigint("5652518929845614012");
                this->p_select_share = bigint("14522637960414365352");
                this->v_select_share = bigint("4689605595382085507");
                this->w_select_share = bigint("7875981768081349499");
            }
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
