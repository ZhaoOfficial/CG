#include <memory>

#include "material.hpp"
#include "ray.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing {
    lambertian::lambertian(const color& albedo) : albedo(albedo) {}

    bool lambertian::scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const {
        vec3 scatter_direction = normal + unit_vector(random_in_unit_sphere());
        // unit_vector(random_in_unit_sphere()) may be opposite normal
        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = normal;
        scattered = ray(hit_point, scatter_direction);
        attenuation = this->albedo;
        return true;
    }

    metal::metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    bool metal::scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const {
        vec3 reflected_direction = reflect(unit_vector(r_in.direction()), normal);
        scattered = ray(hit_point, reflected_direction + this->fuzz * unit_vector(random_in_unit_sphere()));
        attenuation = this->albedo;
        return dot(scattered.direction(), normal) > 0.0f;
    }

    dielectric::dielectric(float index_of_refraction) : refraction_rate(index_of_refraction) {}

    bool dielectric::scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const {
        float refraction_ratio = front_face ? (1.0f / refraction_rate) : refraction_rate;

        vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, normal, refraction_ratio);

        scattered = ray(hit_point, refracted);
        attenuation = color(1.0f, 1.0f, 1.0f);
        return true;
    }

}
