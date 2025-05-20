#include "OT/Row.h"
#include "Tools/Exceptions.h"

template<class T>
bool Row<T>::operator ==(const Row<T>& other) const
{
    return rows == other.rows;
}

template<class T>
Row<T>& Row<T>::operator +=(const Row<T>& other)
{
    if (rows.size() != other.size()) {
        throw invalid_length();
    }
    for (size_t i = 0; i < this->size(); i++)
        rows[i] += other.rows[i];
    return *this;
}

template<class T>
Row<T>& Row<T>::operator -=(const Row<T>& other)
{
    if (rows.size() != other.size()) {
        throw invalid_length();
    }
    for (size_t i = 0; i < this->size(); i++)
        rows[i] -= other.rows[i];
    return *this;
}

template<class T>
Row<T>& Row<T>::operator *=(const T& other)
{
    for (size_t i = 0; i < this->size(); i++)
        rows[i] = rows[i] * other;
    return *this;
}

template<class T>
Row<T> Row<T>::operator *(const T& other)
{
    Row<T> res = *this;
    res *= other;
    return res;
}

template<class T>
DeferredPlus<T, Row<T>> Row<T>::operator +(const Row<T>& other)
{
    return {*this, other};
}

template<class T>
DeferredMinus<T, Row<T>> Row<T>::operator -(const Row<T>& other)
{
    return {*this, other};
}

template<class T>
template<class U>
Row<T>& Row<T>::operator=(const DeferredMinus<T, U>& d)
{
    size_t size = d.x.size();
    rows.resize(size);
    for (size_t i = 0; i < size; i++)
        rows[i] = d[i];
    return *this;
}

template<class T>
template<class U>
Row<T>& Row<T>::operator=(const DeferredPlus<T, U>& d)
{
    size_t size = d.x.size();
    rows.resize(size);
    for (size_t i = 0; i < size; i++)
        rows[i] = d[i];
    return *this;
}

template<class T>
void Row<T>::randomize(PRNG& G, size_t size)
{
	rows.clear();
	rows.reserve(size);
    for (size_t i = 0; i < size; i++)
        rows.push_back(G.get<T>());
}

template<class T>
Row<T> Row<T>::operator<<(int i) const {
    if (i >= T::size() * 8) {
        throw invalid_params();
    }
    Row<T> res = *this;
    for (size_t j = 0; j < this->size(); j++)
        res.rows[j] = res.rows[j] << i;
    return res;
}

template<class T, class U>
void DeferredPlus<T, U>::pack(octetStream& o) const
{
    o.store(this->size());
    for (size_t i = 0; i < this->size(); i++)
        (*this)[i].pack(o);
}

template<class T>
void Row<T>::pack(octetStream& o) const
{
    o.store(this->size());
    for (size_t i = 0; i < this->size(); i++)
        rows[i].pack(o);
}

template<class T>
void Row<T>::unpack(octetStream& o)
{
    size_t size;
    o.get(size);
    rows.clear();
    rows.reserve(size);
    for (size_t i = 0; i < size; i++)
        rows.push_back(o.get<T>());
}

template <class V>
ostream& operator<<(ostream& o, const Row<V>& x)
{
    for (size_t i = 0; i < x.size(); ++i)
        o << x.rows[i] << " | ";
    return o;
}
