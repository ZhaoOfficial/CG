#ifndef _RAY_HPP_
#define _RAY_HPP_

#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class ray {
    public:
        ray(const point3& origin = vec3{ 0.0f,0.0f,0.0f }, const vec3& direction = vec3{ 0.0f,0.0f,0.0f });

        point3 origin() const;
        vec3 direction() const;
        point3 at(float t) const;

    public:
        point3 ori;
        vec3 dir;
    };

    // ray behaviours
    vec3 reflect(const vec3& r_in, const vec3& normal);
    bool reflectance(float cosine, float ref_idx);
    vec3 refract(const vec3& r_in, const vec3& normal, float refraction_ratio);

}

#endif
