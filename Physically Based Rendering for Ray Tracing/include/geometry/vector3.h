#ifndef _PBRT_VECTOR3_H_
#define _PBRT_VECTOR3_H_

#include "tuple3.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
struct Vector3 : public Tuple3<Vector3, T> {
    using value_type      = Tuple3<Vector3, T>::value_type;
    using reference       = Tuple3<Vector3, T>::reference;
    using pointer         = Tuple3<Vector3, T>::pointer;
    using const_reference = Tuple3<Vector3, T>::const_reference;

    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::Tuple3;
};

using Vector3i = Vector3<int>;
using Vector3f = Vector3<Float>;
template <typename T>
using Point3 = Vector3<T>;
using Point3i = Vector3<int>;
using Point3f = Vector3<Float>;
template <typename T>
using Normal3 = Vector3<T>;
using Normal3i = Vector3<int>;
using Normal3f = Vector3<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_VECTOR3_H_
