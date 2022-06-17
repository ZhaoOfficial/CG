#ifndef _PBRT_MATRIX_H_
#define _PBRT_MATRIX_H_

#include "arithmetic.h"
#include "../common.h"

PBRT_NAMESPACE_START

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


PBRT_NAMESPACE_END

#endif // !_PBRT_MATRIX_H_
