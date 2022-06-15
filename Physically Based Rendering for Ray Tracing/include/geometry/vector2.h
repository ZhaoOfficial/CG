#ifndef _PBRT_VECTOR2_H_
#define _PBRT_VECTOR2_H_

#include "tuple2.h"
#include "../common.h"

PBRT_NAMESPACE_START

template <typename T>
struct Vector2 : public Tuple2<Vector2, T> {
    using value_type      = Tuple2<Vector2, T>::value_type;
    using reference       = Tuple2<Vector2, T>::reference;
    using pointer         = Tuple2<Vector2, T>::pointer;
    using const_reference = Tuple2<Vector2, T>::const_reference;

    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;
    using Tuple2<Vector2, T>::Tuple2;
};

using Vector2i = Vector2<int>;
using Vector2f = Vector2<Float>;
template <typename T>
using Point2 = Vector2<T>;
using Point2i = Vector2<int>;
using Point2f = Vector2<Float>;

PBRT_NAMESPACE_END

#endif // !_PBRT_VECTOR2_H_
