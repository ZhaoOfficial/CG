#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_

#include "hittable.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class sphere : public hittable {
    public:
        sphere(point3 center, float radius, std::shared_ptr<material> material_ptr);

        virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec
        ) const override;
        virtual bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const override;

    public:
        point3 center;
        float radius;
        std::shared_ptr<material> material_ptr;
    };

}

#endif // !_SPHERE_HPP_