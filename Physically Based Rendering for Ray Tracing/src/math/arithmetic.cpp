#include "math/arithmetic.h"

PBRT_NAMESPACE_START

constexpr Float deg2rad(Float degree) {
    return degree * FRAC_PI_180;
}

constexpr Float rad2deg(Float radius) {
    return radius * FRAC_180_PI;
}

PBRT_NAMESPACE_END
