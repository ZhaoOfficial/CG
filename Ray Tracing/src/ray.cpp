#include "ray.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {

	ray::ray(const point3& origin, const vec3& direction) : ori(origin), dir(direction) {}

	point3 ray::origin() const { return ori; }
	vec3 ray::direction() const { return dir; }
	point3 ray::at(float t) const { return ori + dir * t; }

    vec3 reflect(const vec3& r_in, const vec3& normal) {
        return r_in - 2.0f * dot(r_in, normal) * normal;
    }

    bool reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return (r0 + (1.0f- r0) * pow((1.0f - cosine), 5)) > gaussian_float(0.5f, 0.4f);
    }

    vec3 refract(const vec3& r_in, const vec3& normal, float refraction_ratio) {
        float cos_theta = std::min(dot(r_in, normal), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        // total internal reflection
        if ((refraction_ratio > 1.0f && refraction_ratio * sin_theta > 1.0f) || reflectance(-cos_theta, refraction_ratio)) {
            return reflect(r_in, normal);
        }
        else {
            vec3 r_out_perp = refraction_ratio * (r_in - cos_theta * normal);
            vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - r_out_perp.length_squared())) * normal;
            return r_out_perp + r_out_parallel;
        }
    }
}
