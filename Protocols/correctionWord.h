#ifndef PROTOCOLS_DABIT_H_
#define PROTOCOLS_DABIT_H_

#include <vector>
using namespace std;

template<class T>
class dpfCorrectionWord
{
public:
    T r_share;
    vector<T> scw;
    vector<T> tcw[2];
    vector<T> output;
};

#endif /* PROTOCOLS_DABIT_H_ */