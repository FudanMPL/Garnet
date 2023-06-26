#ifndef TOOLS_OCT_H_
#define TOOLS_OCT_H_

typedef unsigned char octet;

inline void PRINT_OCTET(const octet* bytes, size_t size) {
    for (size_t i = 0; i < size; ++i)
        cout << hex << (int) bytes[i];
    cout << flush << endl;
}

inline bool OCTETS_EQUAL(const octet* left, const octet* right, int size) {
    for (int i = 0; i < size; ++i)
        if (left[i] != right[i])
            return false;
    return true;
}

#endif
