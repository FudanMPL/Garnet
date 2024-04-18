/*
 * AddableVector.h
 *
 */

#ifndef FHEOFFLINE_ADDABLEVECTOR_H_
#define FHEOFFLINE_ADDABLEVECTOR_H_

#include <vector>
using namespace std;

#include "FHE/Plaintext.h"
#include "Rq_Element.h"

template<class T>
class AddableVector: public vector<T>
{
public:
    AddableVector<T>() {}
    AddableVector<T>(size_t n, const T& x = T()) : vector<T>(n, x) {}
    template <class U, class FD, class S>
    AddableVector<T>(const Plaintext<U,FD,S>& other) :
            AddableVector<T>(other.get_poly()) {}

    template <class U>
    AddableVector<T>(const vector<U>& other)
    {
        this->assign(other.begin(), other.end());
    }

    AddableVector(const Rq_Element& other) :
            AddableVector(other.to_vec_bigint())
    {
    }

    void allocate_slots(const bigint& init)
    {
        for (auto& x: *this)
            x.allocate_slots(init);
    }

    int get_min_alloc()
    {
        int res = 1 << 30;
        for (auto& x: *this)
            res = min(res, x.get_min_alloc());
        return res;
    }

    template <class U>
    AddableVector<T>& operator=(const vector<U>& v)
    {
        this->assign(v.begin(), v.end());
        return *this;
    }

    AddableVector<T>& operator+=(const AddableVector<T>& y)
    {
        if (this->size() != y.size())
            throw out_of_range("vector length mismatch");
        size_t n = this->size();
        for (unsigned int i = 0; i < n; i++)
            (*this)[i] += y[i];
        return *this;
    }

    AddableVector<T> operator-(const AddableVector<T>& y) const
    {
        if (this->size() != y.size())
            throw out_of_range("vector length mismatch");
        AddableVector<T> res;
        res.reserve(y.size());
        size_t n = this->size();
        for (unsigned int i = 0; i < n; i++)
            res.push_back((*this)[i] - y[i]);
        return res;
    }

    void mul(const AddableVector<T>& x, const AddableVector<T>& y)
    {
        if (x.size() != y.size())
            throw length_error("vector length mismatch");
        for (size_t i = 0; i < y.size(); i++)
            (*this)[i].mul(x[i], y[i]);
    }

    AddableVector mul_by_X_i(int i, const FHE_PK& pk) const;

    void generateUniform(PRNG& G, int n_bits)
    {
        for (unsigned int i = 0; i < this->size(); i++)
            (*this)[i].generateUniform(G, n_bits);
    }

    void randomize(PRNG& G)
    {
        for (auto& x : *this)
            x.randomize(G);
    }

    void randomize(PRNG& G, int n_bits, PT_Type type = Evaluation)
    {
        for (auto& x : *this)
            x.randomize(G, n_bits, false, false, type);
    }

    void assign_zero()
    {
        for (auto& x : *this)
            x.assign_zero();
    }

    void assign_one()
    {
        for (auto& x : *this)
            x.assign_one();
    }

    void from(const Generator<T>& generator)
    {
        for (auto& x: *this)
            generator.get(x);
    }

    void pack(octetStream& os) const
    {
        os.store((unsigned int)this->size());
        for (unsigned int i = 0; i < this->size(); i++)
            (*this)[i].pack(os);
    }

    void unpack_size(octetStream& os, const T& init = T())
    {
        unsigned int size;
        os.get(size);
        this->resize(size, init);
    }

    void unpack(octetStream& os, const T& init = T())
    {
        unpack_size(os, init);
        for (unsigned int i = 0; i < this->size(); i++)
            (*this)[i].unpack(os);
    }

    void add(octetStream& os, T& tmp)
    {
        unpack_size(os, tmp);
        T& item = tmp;
        for (unsigned int i = 0; i < this->size(); i++)
        {
            item.unpack(os);
            (*this)[i] += item;
        }
    }

    T infinity_norm() const
    {
        T res = 0;
        for (auto& x: *this)
            res = min(res, x);
        return res;
    }

    size_t report_size(ReportType type)
    {
	size_t res = 4;
	for (unsigned int i = 0; i < this->size(); i++)
	    res += (*this)[i].report_size(type);
	return res;
    }
};

template<class T>
class AddableMatrix: public AddableVector<AddableVector<T> >
{
public:
    void resize(int n)
    {
        this->vector< AddableVector<T> >::resize(n);
    }

    void resize(int n, int m)
    {
        resize(n);
        for (int i = 0; i < n; i++)
            (*this)[i].resize(m);
    }

    AddableMatrix mul_by_X_i(int i, const FHE_PK& pk) const;
};

template<class T>
AddableMatrix<T> AddableMatrix<T>::mul_by_X_i(int i,
        const FHE_PK& pk) const
{
    AddableMatrix<T> res;
    res.resize(this->size());
    for (size_t j = 0; j < this->size(); j++)
        res[j] = (*this)[j].mul_by_X_i(i, pk);
    return res;
}

#endif /* FHEOFFLINE_ADDABLEVECTOR_H_ */
