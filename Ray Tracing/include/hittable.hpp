#ifndef _HITTABLE_HPP_
#define _HITTABLE_HPP_

#include "ray.hpp"
#include "material.hpp"

namespace RayTracing {

    struct hit_record {
        float t{ 0.0f };
        point3 hit_point{ 0.0f, 0.0f, 0.0f };
        vec3 normal{ 0.0f, 0.0f, 0.0f };
        std::shared_ptr<material> material_ptr{ nullptr };
        bool front_face;

        // outward_normal always point outside of the sphere surface
        inline void set_face_normal(const ray& r, const vec3& outward_normal) {
            // dot < 0.0f, the ray is arriving to the sphere
            // dot > 0.0f, the ray is leaving from the sphere
            front_face = (dot(r.direction(), outward_normal) < 0.0f);
            normal = front_face ? outward_normal : -outward_normal;
        }
    };

    class hittable {
    public:
        virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        virtual bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const = 0;
    };

}

#endif // !_HITTABLE_H_