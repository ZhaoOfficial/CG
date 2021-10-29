#ifndef _HITTABLE_HPP_
#define _HITTABLE_HPP_

#include "utility.hpp"
#include "ray.hpp"
#include "material.hpp"

struct hit_record {
    shared_ptr<material> material_ptr;
    point3 p;
    vec3 normal;
    float t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        // dot < 0.0f, not same side
        front_face = r.direction().dot(outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    virtual bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

#endif // !_HITTABLE_H_