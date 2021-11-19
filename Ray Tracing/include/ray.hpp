#ifndef _RAY_HPP_
#define _RAY_HPP_

#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class Ray {
    public:
        Ray(
            const Point3& origin = Vec3{ 0.0f,0.0f,0.0f },
            const Vec3& direction = Vec3{ 0.0f,0.0f,0.0f },
            float time = 0.0f
        );

        Point3 at(float t) const;

    public:
        Point3 origin;
        Vec3 direction;
        float time;
    };

    // ray behaviours
    // normal specular reflect
    Vec3 reflect(const Vec3& r_in, const Vec3& normal);
    // consider polar of ray
    bool reflectance(float cosine, float ref_idx);
    // dielectrics
    Vec3 refract(const Vec3& r_in, const Vec3& normal, float refraction_ratio);

}

#endif // !_RAY_HPP_
