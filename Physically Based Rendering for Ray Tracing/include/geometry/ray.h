#ifndef _PBRT_RAY_H_
#define _PBRT_RAY_H_

#include "vector3.h"
#include "../common.h"

PBRT_NAMESPACE_START

struct Ray {
    Ray(
        Point3f const& o = Point3f{},
        Vector3f const& d = Vector3f{},
        Float t_max = Infinity,
        Float time = Float(0.0),
        Medium const* medium = nullptr
    ) noexcept;
    Point3f operator()(Float t) const;

    friend std::ostream& operator<<(std::ostream& out, Ray const& r);

    // Member variables
    Point3f  o;
    Vector3f d;
    Float mutable t_max;
    Float time;
    Medium const* medium;
};

PBRT_NAMESPACE_END

#endif // !_PBRT_RAY_H_
