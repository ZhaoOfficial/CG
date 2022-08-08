#ifndef _PBRT_DEFINE_H_
#define _PBRT_DEFINE_H_

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

#include "../common.h"

PBRT_NAMESPACE_START

//! Concepts
// Check if the given type is a numeric type.
template <typename T>
concept NumericType = std::is_arithmetic_v<T>;

// Check if the given type is a floating point type.
template <typename T>
concept FloatingPointType = (std::is_floating_point_v<T> and (not std::is_same_v<T, long double>));

// If the given type is `float`, then it returns `uint32_t`.
// If the given type is `double`, then it returns `uint64_t`.
template <FloatingPointType Number>
using FloatingPointBitsType = std::conditional_t<
    std::is_same_v<Number, float>, uint32_t, uint64_t
>;
//! Concepts

//! Common used functions
//* A class contains everything that related to floating point numbers.
struct FloatingPoint {
    template <FloatingPointType Number>
    using FPBT = FloatingPointBitsType<Number>;

    template <FloatingPointType Number>
    static bool close() {static_assert(false); return false;}
    // Convert a floating point number to its bits representation.
    template <FloatingPointType Number>
    static constexpr FPBT<Number> toBits(Number number) {
        return std::bit_cast<FPBT<Number>>(number);
    }
    // Get the sign bit of a floating point.
    template <FloatingPointType Number>
    static constexpr FPBT<Number> getSignBit(Number number) {
        // For `uint32_t`, `digits` is 32.
        // For `uint64_t`, `digits` is 64.
        return toBits(number) bitand (FPBT<Number>(1) << (std::numeric_limits<FPBT<Number>>::digits - 1));
    }
    // Get the exponent part of a floating point.
    template <FloatingPointType Number>
    static constexpr FloatingPointBitsType<Number> getExponent(Number number) {
        // For `float`, `digits` is 24.
        // For `double`, `digits` is 53.
        return toBits(number) bitand (
            (FPBT<Number>(-1) << (std::numeric_limits<Number>::digits - 1)) xor (FPBT<Number>(1) << (std::numeric_limits<FPBT<Number>>::digits - 1))
        );
    }
    // Get the significand part of a floating point.
    template <FloatingPointType Number>
    static constexpr FloatingPointBitsType<Number> getSignificand(Number number) {
        // For `float`, `digits` is 24.
        // For `double`, `digits` is 53.
        return toBits(number) bitand ((FPBT<Number>(1) << (std::numeric_limits<Number>::digits - 1)) - 1);
    }
};

//* About angular.
// Convert from degree form to radius form
constexpr Float deg2rad(Float degree) {
    return degree * FRAC_PI_180;
}

// Convert from radius form to degree form
constexpr Float rad2deg(Float radius) {
    return radius * FRAC_180_PI;
}

// sin(x) / x
constexpr Float sinc(Float x) {
    // lim_{x->0}\sin{x}/x = 1
    if (x * x + Float{1} == Float{1}) {
        return Float{1};
    }
    return std::sin(x) / x;
}

//* Useful arithmetic operations.
// return ad - bc
template <NumericType Number>
constexpr Number crossProductDifference(Number a, Number b, Number c, Number d) {
    return a * d - b * c;
}

// return ad + bc
template <NumericType Number>
constexpr Number crossProductSum(Number a, Number b, Number c, Number d) {
    return a * d + b * c;
}


template <NumericType Number1, NumericType Number2>
constexpr std::common_type_t<Number1, Number2> __mulAddPairCore(Number1 a, Number2 b) {
    return a * b;
}

template <NumericType Number1, NumericType Number2, NumericType... Number>
constexpr std::common_type_t<Number1, Number2, Number...> __mulAddPairCore(Number1 a, Number2 b, Number... numbers) {
    std::common_type_t<Number1, Number2> pair = a * b;
    std::common_type_t<Number...> remain = __mulAddPairCore(numbers...);
    return pair + remain;
}

// return a * b + c * d + ...
template <NumericType... Number>
constexpr std::enable_if_t<
    std::conjunction_v<std::is_arithmetic<Number>...>, std::common_type_t<Number...>
> mulAddPair(Number... numbers) {
    return __mulAddPairCore(numbers...);
}


//! Common used functions

PBRT_NAMESPACE_END

#endif // !_PBRT_DEFINE_H_