/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_UTIL_OPTIONAL_H_
#define TENSORFLOW_SERVING_UTIL_OPTIONAL_H_

#include <array>
#include <functional>
#include <initializer_list>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace serving {
// A value of type tensorflow::serving::optional<T> holds either a value of T or
// an "empty" value.  When it holds a value of T, it stores it as a direct
// subobject, so sizeof(optional<T>) is approximately sizeof(T)+1. The interface
// is based on the upcoming std::experimental::optional<T>, and
// tensorflow::serving::optional<T> is designed to be cheaply drop-in
// replaceable by std::experimental::optional<T>, once it is rolled out.
//
// This implementation is based on the specification in N4335 Section 5:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4335.html
//
// Differences between tensorflow::serving::optional<T> and
// std::experimental::optional<T> include:
//    - tensorflow::serving::optional<T> is basically a proper subset of
//         std::experimental::optional<T>.
//    - constexpr not used. (dependency on some differences between C++11 and
//         C++14.)
//    - noexcept not used.
//    - exceptions not used - in lieu of exceptions we use CHECK-failure.
//
// std::optional<T> might not quite be a drop-in replacement for
// std::experimental::optional<T> because the standards committee is considering
// changes to the semantics of relational operators as part of the process of
// turning std::experimental::optional<T> into std::optional<T>.  The best way
// of making sure you aren't affected by those changes is to make sure that your
// type T defines all of the operators consistently. (x <= y is exactly
// equivalent to !(x > y), etc.)
//
// Synopsis:
//
//     #include "tensorflow_serving/util/optional.h"
//
//     using tensorflow::serving::optional;
//     using tensorflow::serving::nullopt;
//     using tensorflow::serving::in_place;
//     using tensorflow::serving::make_optional;
//
//     optional<string> f() {
//       string result;
//       if (...) {
//          ...
//          result = ...;
//          return result;
//       } else {
//          ...
//          return nullopt;
//       }
//     }
//
//     int main() {
//         optional<string> optstr = f();
//         if (optstr) {
//            // non-empty
//            print(optstr.value());
//         } else {
//            // empty
//            error();
//         }
//     }
template <typename T>
class optional;

// The tag constant `in_place` is used as the first parameter of an optional<T>
// constructor to indicate that the remaining arguments should be forwarded
// to the underlying T constructor.
struct in_place_t {};
extern const in_place_t in_place;

// The tag constant `nullopt` is used to indicate an empty optional<T> in
// certain functions, such as construction or assignment.
struct nullopt_t {
  // It must not be default-constructible to avoid ambiguity for opt = {}.
  explicit constexpr nullopt_t(int /*unused*/) {}
};
extern const nullopt_t nullopt;

// See comment above first declaration.
template <typename T>
class optional {
 public:
  typedef T value_type;

  // A default constructed optional holds the empty value, NOT a default
  // constructed T.
  optional() {}

  // An optional initialized with `nullopt` holds the empty value.
  optional(nullopt_t /*unused*/) {}  // NOLINT(runtime/explicit)

  // Copy constructor, standard semantics.
  optional(const optional& src) {
    if (src) {
      construct(src.reference());
    }
  }

  // Move constructor, standard semantics.
  optional(optional&& src) {  // NOLINT(build/c++11)
    if (src) {
      construct(std::move(src.reference()));
    }
  }

  // Creates a non-empty optional<T> with a copy of the given value of T.
  optional(const T& src) {  // NOLINT(runtime/explicit)
    construct(src);
  }

  // Creates a non-empty optional<T> with a moved-in value of T.
  optional(T&& src) {  // NOLINT
    construct(std::move(src));
  }

  // optional<T>(in_place, arg1, arg2, arg3) constructs a non-empty optional
  // with an in-place constructed value of T(arg1,arg2,arg3).
  template <typename... Args>
  explicit optional(in_place_t /*unused*/,
                    Args&&... args) {  // NOLINT(build/c++11)
    construct(std::forward<Args>(args)...);
  }

  // optional<T>(in_place, {arg1, arg2, arg3}) constructs a non-empty optional
  // with an in-place list-initialized value of T({arg1, arg2, arg3}).
  template <class U, typename... Args>
  explicit optional(in_place_t /*unused*/, std::initializer_list<U> il,
                    Args&&... args) {  // NOLINT(build/c++11)
    construct(il, std::forward<Args>(args)...);
  }

  // Destructor, standard semantics.
  ~optional() { clear(); }

  // Assignment from nullopt: opt = nullopt
  optional& operator=(nullopt_t /*unused*/) {
    clear();
    return *this;
  }

  // Copy assignment, standard semantics.
  optional& operator=(const optional& src) {
    if (src) {
      operator=(src.reference());
    } else {
      clear();
    }
    return *this;
  }

  // Move assignment, standard semantics.
  optional& operator=(optional&& src) {  // NOLINT(build/c++11)
    if (src) {
      operator=(std::move(src.reference()));
    } else {
      clear();
    }
    return *this;
  }

  // Copy assignment from T.  If empty becomes copy construction.
  optional& operator=(const T& src) {  // NOLINT(build/c++11)
    if (*this) {
      reference() = src;
    } else {
      construct(src);
    }
    return *this;
  }

  // Move assignment from T.  If empty becomes move construction.
  optional& operator=(T&& src) {  // NOLINT(build/c++11)
    if (*this) {
      reference() = std::move(src);
    } else {
      construct(std::move(src));
    }
    return *this;
  }

  // Emplace reconstruction.  (Re)constructs the underlying T in-place with the
  // given arguments forwarded:
  //
  // optional<Foo> opt;
  // opt.emplace(arg1,arg2,arg3);  (Constructs Foo(arg1,arg2,arg3))
  //
  // If the optional is non-empty, and the `args` refer to subobjects of the
  // current object, then behaviour is undefined.  This is because the current
  // object will be destructed before the new object is constructed with `args`.
  //
  template <typename... Args>
  void emplace(Args&&... args) {
    clear();
    construct(std::forward<Args>(args)...);
  }

  // Emplace reconstruction with initializer-list.  See immediately above.
  template <class U, class... Args>
  void emplace(std::initializer_list<U> il, Args&&... args) {
    clear();
    construct(il, std::forward<Args>(args)...);
  }

  // Swap, standard semantics.
  void swap(optional& src) {
    if (*this) {
      if (src) {
        using std::swap;
        swap(reference(), src.reference());
      } else {
        src.construct(std::move(reference()));
        destruct();
      }
    } else {
      if (src) {
        construct(std::move(src.reference()));
        src.destruct();
      } else {
        // no effect (swap(disengaged, disengaged))
      }
    }
  }

  // You may use `*opt`, and `opt->m`, to access the underlying T value and T's
  // member `m`, respectively.  If the optional is empty, behaviour is
  // undefined.
  const T* operator->() const {
    DCHECK(engaged_);
    return pointer();
  }
  T* operator->() {
    DCHECK(engaged_);
    return pointer();
  }
  const T& operator*() const {
    DCHECK(engaged_);
    return reference();
  }
  T& operator*() {
    DCHECK(engaged_);
    return reference();
  }

  // In a bool context an optional<T> will return false if and only if it is
  // empty.
  //
  //   if (opt) {
  //     // do something with opt.value();
  //   } else {
  //     // opt is empty
  //   }
  //
  operator bool() const { return engaged_; }

  // Use `opt.value()` to get a reference to underlying value.  The constness
  // and lvalue/rvalue-ness of `opt` is preserved to the view of the T
  // subobject.
  const T& value() const & {
    CHECK(*this) << "Bad optional access";
    return reference();
  }
  T& value() & {
    CHECK(*this) << "Bad optional access";
    return reference();
  }
  T&& value() && {  // NOLINT(build/c++11)
    CHECK(*this) << "Bad optional access";
    return std::move(reference());
  }
  const T&& value() const && {  // NOLINT(build/c++11)
    CHECK(*this) << "Bad optional access";
    return std::move(reference());
  }

  // Use `opt.value_or(val)` to get either the value of T or the given default
  // `val` in the empty case.
  template <class U>
  T value_or(U&& val) const & {
    if (*this) {
      return reference();
    } else {
      return static_cast<T>(std::forward<U>(val));
    }
  }
  template <class U>
  T value_or(U&& val) && {  // NOLINT(build/c++11)
    if (*this) {
      return std::move(reference());
    } else {
      return static_cast<T>(std::forward<U>(val));
    }
  }

 private:
  // Private accessors for internal storage viewed as pointer or reference to T.
  const T* pointer() const {
    return static_cast<const T*>(static_cast<const void*>(&storage_));
  }
  T* pointer() { return static_cast<T*>(static_cast<void*>(&storage_)); }
  const T& reference() const { return *pointer(); }
  T& reference() { return *pointer(); }

  // Construct inner T in place with given `args`.
  // Precondition: engaged_ is false
  // Postcondition: engaged_ is true
  template <class... Args>
  void construct(Args&&... args) {
    DCHECK(!engaged_);
    engaged_ = true;
    new (pointer()) T(std::forward<Args>(args)...);
    DCHECK(engaged_);
  }

  // Destruct inner T.
  // Precondition: engaged_ is true
  // Postcondition: engaged_ is false
  void destruct() {
    DCHECK(engaged_);
    pointer()->T::~T();
    engaged_ = false;
    DCHECK(!engaged_);
  }

  // Destroy inner T if engaged.
  // Postcondition: engaged_ is false
  void clear() {
    if (engaged_) {
      destruct();
    }
    DCHECK(!engaged_);
  }

  // The internal storage for a would-be T value, constructed and destroyed
  // with placement new and placement delete.
  typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;

  // Whether or not this optional is non-empty.
  bool engaged_ = false;

  // T constraint checks. You can't have an optional of nullopt_t, in_place_t or
  // a reference.
  static_assert(
      !std::is_same<nullopt_t, typename std::remove_cv<T>::type>::value,
      "optional<nullopt_t> is not allowed.");
  static_assert(
      !std::is_same<in_place_t, typename std::remove_cv<T>::type>::value,
      "optional<in_place_t> is not allowed.");
  static_assert(!std::is_reference<T>::value,
                "optional<reference> is not allowed.");
};

// make_optional(v) creates a non-empty optional<T> where the type T is deduced
// from v.  Can also be explicitly instantiated as make_optional<T>(v).
template <typename T>
optional<typename std::decay<T>::type> make_optional(T&& v) {
  return optional<typename std::decay<T>::type>(std::forward<T>(v));
}

// All combinations of the six comparisons between optional<T>, T and nullopt.
// The empty value is considered less than all non-empty values, and equal to
// itself.
template <typename T>
bool operator==(const optional<T>& lhs, const optional<T>& rhs) {
  if (lhs) {
    if (rhs) {
      return *lhs == *rhs;
    } else {
      return false;
    }
  } else {
    if (rhs) {
      return false;
    } else {
      return true;
    }
  }
}

template <typename T>
bool operator!=(const optional<T>& lhs, const optional<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
bool operator<(const optional<T>& lhs, const optional<T>& rhs) {
  if (!rhs) {
    return false;
  } else if (!lhs) {
    return true;
  } else {
    return *lhs < *rhs;
  }
}

template <typename T>
bool operator>(const optional<T>& lhs, const optional<T>& rhs) {
  return rhs < lhs;
}

template <typename T>
bool operator<=(const optional<T>& lhs, const optional<T>& rhs) {
  return !(rhs < lhs);
}

template <typename T>
bool operator>=(const optional<T>& lhs, const optional<T>& rhs) {
  return !(lhs < rhs);
}

template <typename T>
bool operator==(const optional<T>& lhs, nullopt_t rhs) {
  return !lhs;
}

template <typename T>
bool operator==(nullopt_t lhs, const optional<T>& rhs) {
  return !rhs;
}

template <typename T>
bool operator!=(const optional<T>& lhs, nullopt_t rhs) {
  return static_cast<bool>(lhs);
}

template <typename T>
bool operator!=(nullopt_t lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs);
}

template <typename T>
bool operator<(const optional<T>& lhs, nullopt_t rhs) {
  return false;
}

template <typename T>
bool operator<(nullopt_t lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs);
}

template <typename T>
bool operator<=(const optional<T>& lhs, nullopt_t rhs) {
  return !lhs;
}

template <typename T>
bool operator<=(nullopt_t lhs, const optional<T>& rhs) {
  return true;
}

template <typename T>
bool operator>(const optional<T>& lhs, nullopt_t rhs) {
  return static_cast<bool>(lhs);
}

template <typename T>
bool operator>(nullopt_t lhs, const optional<T>& rhs) {
  return false;
}

template <typename T>
bool operator>=(const optional<T>& lhs, nullopt_t rhs) {
  return true;
}

template <typename T>
bool operator>=(nullopt_t lhs, const optional<T>& rhs) {
  return !rhs;
}

template <typename T>
bool operator==(const optional<T>& lhs, const T& rhs) {
  return static_cast<bool>(lhs) ? *lhs == rhs : false;
}

template <typename T>
bool operator==(const T& lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs) ? lhs == *rhs : false;
}

template <typename T>
bool operator!=(const optional<T>& lhs, const T& rhs) {
  return static_cast<bool>(lhs) ? !(*lhs == rhs) : true;
}

template <typename T>
bool operator!=(const T& lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs) ? !(lhs == *rhs) : true;
}

template <typename T>
bool operator<(const optional<T>& lhs, const T& rhs) {
  return static_cast<bool>(lhs) ? *lhs < rhs : true;
}

template <typename T>
bool operator<(const T& lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs) ? lhs < *rhs : false;
}

template <typename T>
bool operator>(const optional<T>& lhs, const T& rhs) {
  return static_cast<bool>(lhs) ? rhs < *lhs : false;
}

template <typename T>
bool operator>(const T& lhs, const optional<T>& rhs) {
  return static_cast<bool>(rhs) ? *rhs < lhs : true;
}

template <typename T>
bool operator<=(const optional<T>& lhs, const T& rhs) {
  return !(lhs > rhs);
}

template <typename T>
bool operator<=(const T& lhs, const optional<T>& rhs) {
  return !(lhs > rhs);
}

template <typename T>
bool operator>=(const optional<T>& lhs, const T& rhs) {
  return !(lhs < rhs);
}

template <typename T>
bool operator>=(const T& lhs, const optional<T>& rhs) {
  return !(lhs < rhs);
}

// Swap, standard semantics.
template <typename T>
inline void swap(optional<T>& a, optional<T>& b) {
  a.swap(b);
}

}  // end namespace serving
}  // end namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_OPTIONAL_H_
