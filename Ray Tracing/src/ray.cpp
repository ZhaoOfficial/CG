#include "ray.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {

	Ray::Ray(const Point3& origin, const Vec3& direction, float time) : origin(origin), direction(unit_vector(direction)), time(time) {}

	Point3 Ray::at(float t) const { return origin + direction * t; }

    Vec3 reflect(const Vec3& r_in, const Vec3& normal) {
        return r_in - 2.0f * dot(r_in, normal) * normal;
    }

    bool reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return (r0 + (1.0f- r0) * pow((1.0f - cosine), 5)) > gaussian_float(0.5f, 0.4f);
    }

    Vec3 refract(const Vec3& r_in, const Vec3& normal, float refraction_ratio) {
        float cos_theta = std::min(dot(r_in, normal), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        // total internal reflection
        if ((refraction_ratio > 1.0f && refraction_ratio * sin_theta > 1.0f) || reflectance(-cos_theta, refraction_ratio)) {
            return reflect(r_in, normal);
        }
        else {
            Vec3 r_out_perp = refraction_ratio * (r_in - cos_theta * normal);
            Vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - r_out_perp.length_squared())) * normal;
            return r_out_perp + r_out_parallel;
        }
    }
}
