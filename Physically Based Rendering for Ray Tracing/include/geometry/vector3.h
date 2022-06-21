#ifndef _PBRT_VECTOR3_H_
#define _PBRT_VECTOR3_H_

#include <array>

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

// Return the cross-product of two `Vector3`.
template <typename T, typename U>
Vector3<decltype(T{} * U{})> cross(Vector3<T> const& lhs, Vector3<U> const& rhs) {
    return Vector3<decltype(T{} * U{})> {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

// The volume of parallelepiped.
// If the three edge vector are `a`, `b`, and `c`,
// then the volume is `dot(cross(a, b), c)`.
// Note that the volume may be negative.
// 5 add, 9 multiply.
template<NumericType Number>
constexpr Number volumeOfParallelepiped(Vector3<Number> const& a, Vector3<Number> const& b, Vector3<Number> const& c) {
    return dot(cross(a, b), c);
}

template<NumericType Number>
constexpr Number volumeOfParallelepiped(std::array<Number, 9> const& arr) {
    return volumeOfParallelepiped(
        Vector3<Number>{ arr[0], arr[1], arr[2] },
        Vector3<Number>{ arr[3], arr[4], arr[5] },
        Vector3<Number>{ arr[6], arr[7], arr[8] }
    );
}

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
