/*
 * aby2Share.h
 *
 */
#ifndef PROTOCOLS_ABY2SHARE_H_
#define PROTOCOLS_ABY2SHARE_H_

#include "SemiShare.h"
#include "Semi.h"
#include "OT/Rectangle.h"
#include "GC/SemiSecret.h"



#include "Math/FixedVec.h"
#include "Math/Integer.h"
#include "Protocols/aby2ed.h"
#include "Protocols/Rep3Shuffler.h"
#include "GC/ShareSecret.h"
#include "ShareInterface.h"
#include "Processor/Instruction.h"
#include "aby2Input.h"

template<class T> class ReplicatedPrep;
template<class T> class SemiRep3Prep;
template<class T> class ReplicatedRingPrep;
template<class T> class ReplicatedPO;
template<class T> class SpecificPrivateOutput;

template<int K>//这里的T是Z2<K>
class aby2Share : public SemiShare<SignedZ2<K>>
{
    typedef Z2<K> T;
    // typedef aby2Share This;
    // typedef FixedVec<T, 2> super;
    // typedef ShareInterface super1;

public:
    typedef SemiMC<aby2Share> MAC_Check;
    typedef DirectSemiMC<aby2Share> Direct_MC;
    typedef SemiInput<aby2Share> Input;
    typedef ::PrivateOutput<aby2Share> PrivateOutput;
    typedef Semi<aby2Share> Protocol;
    typedef SemiPrep2k<aby2Share> LivePrep;

    typedef aby2Share prep_type;
    typedef SemiMultiplier<aby2Share> Multiplier;
    typedef OTTripleGenerator<prep_type> TripleGenerator;
    typedef Z2kSquare<K> Rectangle;
    static const bool has_split = true;



    // // typedef T clear; //和下面的redefine了
    // typedef T open_type;
    // typedef T element_type;

    // typedef aby2ed<aby2Share> Protocol;
    // // typedef ReplicatedMC<aby2Share> MAC_Check;
    // typedef ReplicatedMC<aby2Share> MAC_Check;
    // typedef MAC_Check Direct_MC;
    // typedef aby2Input<aby2Share> Input;
    // typedef ReplicatedPO<This> PO;
    // typedef SpecificPrivateOutput<This> PrivateOutput;
    // typedef SemiRep3Prep<aby2Share> LivePrep;
    // typedef aby2Share Honest;
    // typedef SignedZ2<K> clear;
    // // typedef FixedVec<Z2<K>,2> clear;

    // typedef GC::SemiHonestRepSecret bit_type;       

    // const static bool needs_ot = false;
    // const static bool dishonest_majority = false;
    // const static bool expensive = false;

    // const static bool variable_players = false;
    // static const bool has_trunc_pr = true;
    // static const bool malicious = false;

    // static aby2Share constant(T value, int my_num, typename super1::mac_key_type = {})
    // {
    //     return aby2Share(value, my_num);
    // }

    aby2Share()
    {
        // printf("进入aby2Share()\n");
    }

    template<class U>
    aby2Share(const U& other) :SemiShare<SignedZ2<K>>(other)
    {
    }
    
    aby2Share(const T&other, int my_num, const T& alphai = {})
    {
        printf("进入aby2Share(const T&other, int my_num, const T& alphai = {}) num:%d\n",my_num);
        std::cout<<other<<std::endl;
        (void) alphai;
        assign(other, my_num);
        // (void) alphai;
        // printf("进入aby2Share(T value, int my_num, const T& alphai = {}))构造函数\n");
        // // Replicated<aby2Share>::assign(*this, value, my_num);
        // // aby2ed<aby2Share>::assign(*this, value, my_num);
        // assert(This::vector_length == 2);
        // auto share=*this;
        // share.assign_zero();
        // // if (my_num < 2)
        // //     share[my_num] = value;
        // T delta_0=100;
        // T delta_1=111;
        // share[0]=delta_0+delta_1+value;// Delta暂时硬编码成100+111+value
        // if (my_num ==0)
        //     share[1]=delta_0;
        // else
        //     share[1]=delta_1;
    }

template<class U>
    static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
            const aby2Share* source, int n_inputs,
            typename U::Protocol& protocol)
    {
        auto& P = protocol.P;
        int my_num = P.my_num();
        int unit = GC::Clear::N_BITS;
        for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
        {
            int start = k * unit;
            int m = min(unit, n_inputs - start);
            int n = regs.size() / n_bits;
            if (P.num_players() != n)
                throw runtime_error(
                        to_string(n) + "-way split not working with "
                                + to_string(P.num_players()) + " parties");

            for (int l = 0; l < n_bits; l += unit)
            {
                int base = l;
                int n_left = min(n_bits - base, unit);
                for (int i = base; i < base + n_left; i++)
                    for (int j = 0; j < n; j++)
                        dest.at(regs.at(n * i + j) + k) = {};

                square64 square;

                for (int j = 0; j < m; j++)
                    square.rows[j] = source[j + start].get_limb(l / unit);

                square.transpose(m, n_left);

                for (int j = 0; j < n_left; j++)
                {
                    auto& dest_reg = dest.at(
                            regs.at(n * (base + j) + my_num) + k);
                    dest_reg = square.rows[j];
                }
            }
        }
    }



    // static string type_short()
    // {
    //     return "R" + string(1, clear::type_char());
    // }
    // static int threshold(int)
    // {
    //     return 1;
    // }

    // static void specification(octetStream& os)
    // {
    //     T::specification(os);
    // }

    // void pack(octetStream& os, T) const
    // {
    //     pack(os, false);
    // }
    // void pack(octetStream& os, bool full = true) const
    // {
    //     if (full)
    //         FixedVec<T, 2>::pack(os);
    //     else
    //         (*this)[0].pack(os);
    // }
    // void unpack(octetStream& os, bool full = true)
    // {
    //     assert(full);
    //     FixedVec<T, 2>::unpack(os);
    // }

    // template<class U>
    // static void shrsi(SubProcessor<U>& proc, const Instruction& inst)
    // {
    //     shrsi(proc, inst, T::prime_field);
    // }

    // template<class U>
    // static void shrsi(SubProcessor<U>&, const Instruction&,
    //         true_type)
    // {
    //     throw runtime_error("shrsi not implemented");
    // }

    // template<class U>
    // static void shrsi(SubProcessor<U>& proc, const Instruction& inst,
    //         false_type)
    // {
    //     for (int i = 0; i < inst.get_size(); i++)
    //     {
    //         auto& dest = proc.get_S_ref(inst.get_r(0) + i);
    //         auto& source = proc.get_S_ref(inst.get_r(1) + i);
    //         dest = source >> inst.get_n();
    //     }
    // }


    // clear local_mul(const aby2Share& other) const
    // {
    //     auto a = (*this)[0].lazy_mul(other.lazy_sum());
    //     auto b = (*this)[1].lazy_mul(other[0]);
    //     return a.lazy_add(b);  
    // }
};



#endif /* PROTOCOLS_ABY2SHARE_H_ */