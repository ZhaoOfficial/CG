#ifndef _PBRT_SWIZZLE_H_
#define _PBRT_SWIZZLE_H_

#include <type_traits>

#include "../common.h"

PBRT_NAMESPACE_START

template <typename T, std::size_t N, std::size_t... Indices>
class Swizzle {
    // Interface of vector.
    using value_type      = std::decay_t<T>;
    using reference       = value_type&;
    using pointer         = value_type*;
    using const_reference = value_type const&;

    constexpr static std::size_t indices[] = { Indices... };

    const_reference operator[](std::size_t index) const {
        return this->data[indices[index]];
    }
    reference operator[](std::size_t index) {
        return this->data[indices[index]];
    }

    T data[N];
};

PBRT_NAMESPACE_END

#endif // !_PBRT_SWIZZLE_H_
