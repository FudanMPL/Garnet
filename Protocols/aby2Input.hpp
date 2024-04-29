/*
 * aby2Input.cpp
 *
 */

#ifndef PROTOCOLS_ABY2SHARE_HPP_
#define PROTOCOLS_ABY2SHARE_HPP_

#include "aby2Input.h"
#include "Processor/Processor.h"

#include "Processor/Input.hpp"

template<class T>//这里的T是Rep3Share2
void aby2Input<T>::reset(int player)
{
    // InputBase<T>::reset(player);//这里的reset是一个非静态成员函数，不能这样调用吧？
    assert(P.num_players() == 2);
    if (player == P.my_num())
    {
        this->shares.clear();
        os.resize(2);
        for (auto& o : os)
            o.reset_write_head();
    }
    expect[player] = false;
}

template<class T>  
inline void aby2Input<T>::add_mine(const typename T::open_type& input, int n_bits)
{
    cout<<"进入aby2Input<T>::add_mine函数"<<endl;
    auto& shares = this->shares;//是PointerVectorI10Rep3Share2ILi64EEE
    shares.push_back({});
    T& my_share = shares.back();//shares.back()是Rep3Share2ILi64EE类型
    my_share[0].randomize(protocol.shared_prngs[0], n_bits);//protocol.shared_prngs[0]为双方共同拥有的随机种子
    my_share[1].randomize(protocol.shared_prngs[1], n_bits);//这里的my_share是2Z2ILi64EE类型
    my_share[0]=my_share[0]+my_share[1]+input;
    my_share[0].pack(os[P.my_num()], n_bits);
    cout<<"在aby2Input<T>::add_mine函数中，my_share数组为："<<my_share[0]<<'\t'<<my_share[1]<<endl; //此时share为（Delta,[delta])
    this->values_input++;
    expect[P.my_num()] = true;
}

template<class T>
void aby2Input<T>::add_other(int player, int)
{
    cout<<"进入add_other："<<player<<endl;
    expect[player] = true;
}

template<class T>
void aby2Input<T>::send_mine()
{
    P.send_relative(os);
}

template<class T>
void aby2Input<T>::exchange()//发送与接收存在os中数据流
{
    cout<<"进入exchange函数"<<endl;
    // for(auto i:expect) cout<<'\t'<<i;
    int mynum=P.my_num();
    bool send = expect[mynum];
    
    // cout<<'\n'<<send<<endl;
    if(send)//当前P为拥有x的一方
    {
        // cout<<"发送方"<<endl;
        P.send_to(1-mynum,os[mynum]);
    }
    else
    {
        // cout<<"接收方"<<endl;
        auto& dest =  InputBase<T>::os[1-mynum];//因为后面的finalize_other函数中octeStream参数为os[player]其中player为拥有x的一方
        P.receive_player(1-mynum , dest);
        // cout<<"接收方结束"<<endl;
    }
}

template<class T>
inline void aby2Input<T>::finalize_other(int player, T& target,octetStream& o, int n_bits)//处理os中的数据流
{
    typename T::value_type t;//这里的t类型为Z2<K>
    t.unpack(o, n_bits);
    target[0]=t;
    target[1].randomize(protocol.shared_prngs[0], n_bits);
    cout<<"在aby2Input<T>::finalize_other函数中，target数组为：\n"<<target[0]<<'\t'<<target[1]<<endl;
}


#endif
