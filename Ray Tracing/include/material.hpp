#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include "ray.hpp"
#include "utility.hpp"
#include "vec3.hpp"

namespace RayTracing  {

    class material {
    public:
        virtual bool scatter(
            const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
        ) const = 0;
    };

    class lambertian : public material {
    public:
        lambertian(const color& albedo);

        virtual bool scatter(
            const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
        ) const override;

    public:
        color albedo;
    };

    class metal : public material {
    public:
        metal(const color& albedo, float fuzz);

        virtual bool scatter(
            const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
        ) const override;

    public:
        color albedo;
        float fuzz;
    };

    class dielectric : public material {
    public:
        dielectric(float index_of_refraction);

        virtual bool scatter(
            const ray& r_in, const vec3& hit_point, const vec3& normal, bool front_face, color& attenuation, ray& scattered
        ) const override;

    public:
        float refraction_rate;
    };

}

#endif // !_MATERIAL_HPP_