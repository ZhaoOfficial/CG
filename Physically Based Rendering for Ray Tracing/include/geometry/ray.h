#ifndef _PBRT_RAY_H_
#define _PBRT_RAY_H_

#include <iostream>

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
    ) noexcept : o{o}, d{d}, t_max{t_max}, time{time}, medium{medium} {}
    Point3f operator()(Float t) const { return o + d * t; }

    friend std::ostream& operator<<(std::ostream& out, Ray const& r) {
        out << "[o = " << r.o << ", d = " << r.d << ", t_max = " << r.t_max << ", time = " << r.time << "]";
        return out;
    }

    // Member variables
    Point3f  o;
    Vector3f d;
    Float mutable t_max;
    Float time;
    Medium const* medium;
};

PBRT_NAMESPACE_END

#endif // !_PBRT_RAY_H_
