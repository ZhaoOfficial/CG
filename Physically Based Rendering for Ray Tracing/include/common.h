#ifndef _PBRT_COMMON_H_
#define _PBRT_COMMON_H_

#include <limits>

#define PBRT_NAMESPACE_START namespace pbrt {
#define PBRT_NAMESPACE_END   }

PBRT_NAMESPACE_START

using Float = float;
// using Float = double;

//! Constants
constexpr Float MinFloat  = std::numeric_limits<Float>::lowest();
constexpr Float MaxFloat  = std::numeric_limits<Float>::max();
constexpr Float Infinity  = std::numeric_limits<Float>::infinity();
constexpr Float Epsilon   = std::numeric_limits<Float>::epsilon();
constexpr Float PI        = 3.14159265358979323846;
constexpr Float INV_PI    = 0.31830988618379067154;
constexpr Float INV_2PI   = 0.15915494309189533577;
constexpr Float INV_4PI   = 0.07957747154594766788;
constexpr Float PI_OVER_2 = 1.57079632679489661923;
constexpr Float PI_OVER_4 = 0.78539816339744830961;
constexpr Float SQRT_2    = 1.41421356237309504880;
//! Constants

//! Declarations of a lot of classes
class Medium;
//! Declarations of a lot of classes

PBRT_NAMESPACE_END

#endif // !_PBRT_COMMON_H_
