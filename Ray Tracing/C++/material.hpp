#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include "utility.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include <iostream>

vec3 reflect(const vec3& r_in, const vec3& normal) {
    return r_in - 2.0f * r_in.dot(normal) * normal;
}

bool reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5) > gaussian_float(0.5f, 0.4f);
}

vec3 refract(const vec3& r_in, const vec3& normal, float refraction_ratio) {
    float cos_theta = r_in.dot(normal);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    
    if (refraction_ratio * sin_theta > 1.0f || reflectance(-cos_theta, refraction_ratio)){
        return reflect(r_in, normal);
    }
    else {
        vec3 r_out_perp = refraction_ratio * (r_in - cos_theta * normal);
        vec3 r_out_parallel = -sqrt(1.0f - r_out_perp.length_squared()) * normal;
        return r_out_perp + r_out_parallel;
    }
}

class material {
public:
    virtual bool scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const = 0;
};

class lambertian : public material {
public:
    lambertian(const color& a) : albedo(a) {}

    virtual bool scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const override {
        vec3 scatter_direction = normal + vec3::random_in_unit_sphere();
        if (scatter_direction.near_zero())
            scatter_direction = normal;
        scattered = ray(hit_point, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

class metal : public material {
public:
    metal(const color& a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

    virtual bool scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const override {
        vec3 reflected = reflect(r_in.direction().unit_vector(), normal);
        scattered = ray(hit_point, reflected + fuzz * vec3::random_in_unit_sphere());
        attenuation = albedo;
        return scattered.direction().dot(normal) > 0.0f;
    }

public:
    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    dielectric(float index_of_refraction) : refraction_rate(index_of_refraction) {}

    virtual bool scatter(
        const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
    ) const override {
        float refraction_ratio = front_face ? (1.0f / refraction_rate) : refraction_rate;

        vec3 unit_direction = r_in.direction().unit_vector();
        vec3 refracted = refract(unit_direction, normal, refraction_ratio);

        scattered = ray(hit_point, refracted);
        attenuation = color(1.0f, 1.0f, 1.0f);
        return true;
    }

public:
    float refraction_rate;
};

#endif // !_MATERIAL_HPP_