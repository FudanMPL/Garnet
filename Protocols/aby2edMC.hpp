// /*
//  * aby2edMC.cpp
//  *
//  */

// #ifndef PROTOCOLS_aby2edMC_HPP_
// #define PROTOCOLS_aby2edMC_HPP_

// #include "aby2edMC.h"

// template<class T>
// void aby2edMC<T>::POpen(vector<typename T::open_type>& values,
//         const vector<T>& S, const Player& P)
// {
//     prepare(S);
//     P.pass_around(to_send, o, -1);
//     finalize(values, S);
// }

// template<class T>
// void aby2edMC<T>::POpen_Begin(vector<typename T::open_type>&,
//         const vector<T>& S, const Player& P)
// {
//     prepare(S);
//     P.send_relative(-1, to_send);
// }

// template<class T>
// void aby2edMC<T>::prepare(const vector<T>& S)
// {
//     assert(T::vector_length == 2);
//     o.reset_write_head();
//     to_send.reset_write_head();
//     to_send.reserve(S.size() * T::value_type::size());
//     for (auto& x : S)
//         x[1].pack(to_send);//这里我修改了由x[0]-->x[1]
// }

// template<class T>
// void aby2edMC<T>::exchange(const Player& P)
// {
//     prepare(this->secrets);
//     int mynum=P.my_num();
//     P.send_to(1-mynum,to_send);
//     P.receive_player(1-mynum, o)

//     // P.pass_around(to_send, o, -1);
// }

// template<class T>
// void aby2edMC<T>::POpen_End(vector<typename T::open_type>& values,
//         const vector<T>& S, const Player& P)
// {
//     P.receive_relative(1, o);
//     finalize(values, S);
// }

// template<class T>
// void aby2edMC<T>::finalize(vector<typename T::open_type>& values,
//         const vector<T>& S)
// {
//     values.resize(S.size());
//     for (size_t i = 0; i < S.size(); i++)
//     {
//         typename T::open_type tmp;
//         tmp.unpack(o);
//         values[i] = S[i].sum() + tmp;
//     }
// }

// template<class T>
// typename T::open_type aby2edMC<T>::finalize_raw()
// {
//     // auto a = this->secrets.next().sum();
//     return this->secrets[0]-this->secrets[1]-o.get<typename T::open_type>();
// }

// #endif
