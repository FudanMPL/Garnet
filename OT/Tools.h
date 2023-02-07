#ifndef _OTTOOLS
#define _OTTOOLS

//#if defined(__SSE2__)
/*
 * Convert __m128i to string of type T
 */
template <typename T>
string __m128i_toString(const __m128i var) {
    stringstream sstr;
    sstr << hex;
    const T* values = (const T*) &var;
    if (sizeof(T) == 1) {
        for (unsigned int i = 0; i < sizeof(__m128i); i++) {
            sstr << (int) values[i] << " ";
        }
    } else {
        for (unsigned int i = 0; i < sizeof(__m128i) / sizeof(T); i++) {
            sstr << values[i] << " ";
        }
    }
    return sstr.str();
}
//#endif

#endif
