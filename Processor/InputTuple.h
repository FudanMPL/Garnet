/*
 * InputTuple.h
 *
 */

#ifndef PROCESSOR_INPUTTUPLE_H_
#define PROCESSOR_INPUTTUPLE_H_


template <class T>
struct InputTuple
{
  T share;
  typename T::open_type value;

  static int size()
    { return T::open_type::size() + T::size(); }

  static string type_string()
    { return T::type_string(); }

  static void specification(octetStream& os)
    {
      T::specification(os);
    }

  InputTuple() {}

  InputTuple(const T& share, const typename T::open_type& value) : share(share), value(value) {}

  template<class U>
  InputTuple(const InputTuple<U>& other) : share(other.share), value(other.value) {}

  void assign(const char* buffer)
    {
      share.assign(buffer);
      value.assign(buffer + T::size());
    }
};


template <class T>
struct RefInputTuple
{
  T& share;
  typename T::open_type& value;
  RefInputTuple(T& share, typename T::open_type& value) : share(share), value(value) {}
  void operator=(InputTuple<T>& other) { share = other.share; value = other.value; }
};


#endif /* PROCESSOR_INPUTTUPLE_H_ */
