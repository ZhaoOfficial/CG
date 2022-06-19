#ifndef _PBRT_DEFINE_H_
#define _PBRT_DEFINE_H_

#include <type_traits>

#include "../common.h"

PBRT_NAMESPACE_START

//! Concepts
template <typename T>
concept NumericType = std::is_arithmetic_v<T>;
//! Concepts

//! Common used functions
// Convert from degree form to radius form
inline constexpr Float deg2rad(Float degree) {
    return degree * FRAC_PI_180;
}

// Convert from radius form to degree form
inline constexpr Float rad2deg(Float radius) {
    return radius * FRAC_180_PI;
}

// return ad - bc
template<NumericType Number>
constexpr Number crossProductDifference(Number a, Number b, Number c, Number d) {
    return a * d - b * c;
}

// return ad + bc
template<NumericType Number>
constexpr Number crossProductSum(Number a, Number b, Number c, Number d) {
    return a * d + b * c;
}

//! Common used functions

PBRT_NAMESPACE_END

#endif // !_PBRT_DEFINE_H_