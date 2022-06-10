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
//! Common used functions

PBRT_NAMESPACE_END

#endif // !_PBRT_DEFINE_H_