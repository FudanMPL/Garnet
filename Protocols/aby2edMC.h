// /*
//  * aby2edMC.h
//  *
//  */

// #ifndef PROTOCOLS_aby2edMC_H_
// #define PROTOCOLS_aby2edMC_H_

// #include "MAC_Check_Base.h"

// /**
//  * Replicated semi-honest three-party opening protocol
//  */
// template <class T>
// class aby2edMC : public MAC_Check_Base<T>
// {
//     octetStream o;
//     octetStream to_send;

//     void prepare(const vector<T>& S);
//     void finalize(vector<typename T::open_type>& values, const vector<T>& S);

// public:
//     // emulate MAC_Check
//     aby2edMC(const typename T::mac_key_type& _ = {}, int __ = 0, int ___ = 0)
//     { (void)_; (void)__; (void)___; }

//     // emulate Direct_MAC_Check
//     aby2edMC(const typename T::mac_key_type& _, Names& ____, int __ = 0, int ___ = 0)
//     { (void)_; (void)__; (void)___; (void)____; }

//     void POpen(vector<typename T::open_type>& values,const vector<T>& S,const Player& P);
//     void POpen_Begin(vector<typename T::open_type>& values,const vector<T>& S,const Player& P);
//     void POpen_End(vector<typename T::open_type>& values,const vector<T>& S,const Player& P);

//     virtual void exchange(const Player& P);
//     virtual typename T::open_type finalize_raw();

//     void Check(const Player& P) { (void)P; }

//     aby2edMC& get_part_MC()
//     {
//         return *this;
//     }
// };

// #endif /* PROTOCOLS_aby2edMC_H_ */
