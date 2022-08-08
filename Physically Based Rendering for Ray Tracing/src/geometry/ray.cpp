#include <fmt/ranges.h>
#include "geometry/ray.h"

PBRT_NAMESPACE_START

Ray::Ray(
    Point3f const& o,
    Vector3f const& d,
    Float t_max,
    Float time,
    Medium const* medium
) noexcept : o{o}, d{d}, t_max{t_max}, time{time}, medium{medium} {}

Point3f Ray::operator()(Float t) const {
    return o + d * t;
}

std::ostream& operator<<(std::ostream& out, Ray const& r) {
    out << "[o = " << r.o << ", d = " << r.d 
        << fmt::format(", t_max = {}, time = {}]", r.t_max, r.time);
    return out;
}

PBRT_NAMESPACE_END
