#ifndef _RAY_HPP_
#define _RAY_HPP_

#include "vec3.hpp"

class ray {
    public:
        ray() {}
        ray(const point3& origin, const vec3& direction)
            : ori(origin), dir(direction)
        {}

        point3 origin() const  { return ori; }
        vec3 direction() const { return dir; }

        point3 at(float t) const {
            return ori + dir * t;
        }

    public:
        point3 ori;
        vec3 dir;
};

#endif
