#ifndef _PBRT_COMMON_H_
#define _PBRT_COMMON_H_

#include <iostream>
#include <limits>

#define PBRT_NAMESPACE_START namespace pbrt {
#define PBRT_NAMESPACE_END   }

PBRT_NAMESPACE_START

using Float = float;
// using Float = double;

//! Constants
constexpr Float MinFloat    = std::numeric_limits<Float>::lowest();
constexpr Float MaxFloat    = std::numeric_limits<Float>::max();
constexpr Float Infinity    = std::numeric_limits<Float>::infinity();
constexpr Float Epsilon     = std::numeric_limits<Float>::epsilon();
constexpr Float SoftEps     = Float(1e-5);
constexpr Float PI          = 3.14159265358979323846l;
constexpr Float FRAC_1_PI   = 0.31830988618379067154l;
constexpr Float FRAC_1_2PI  = 0.15915494309189533577l;
constexpr Float FRAC_1_4PI  = 0.07957747154594766788l;
constexpr Float FRAC_PI_2   = 1.57079632679489661923l;
constexpr Float FRAC_PI_4   = 0.78539816339744830961l;
constexpr Float FRAC_PI_180 = 0.01745329251994329577l;
constexpr Float FRAC_180_PI = 57.2957795130823208768l;
constexpr Float SQRT_2      = 1.41421356237309504880l;
constexpr Float SQRT_3      = 1.73205080756887729353l;
//! Constants

//! Declarations of a lot of classes
class Medium;
//! Declarations of a lot of classes

PBRT_NAMESPACE_END

#endif // !_PBRT_COMMON_H_
