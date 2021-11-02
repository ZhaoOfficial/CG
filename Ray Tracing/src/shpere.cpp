#include <memory>

#include "hittable.hpp"
#include "sphere.hpp"
#include "vec3.hpp"

namespace RayTracing {

    sphere::sphere(point3 center, float radius, std::shared_ptr<material> material_ptr) : center(center), radius(radius), material_ptr(material_ptr) {};

    bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        vec3 oc = r.origin() - this->center;
        float a = r.direction().length_squared();
        float b = dot(r.direction(), oc);
        float c = oc.length_squared() - this->radius * this->radius;

        float delta = b * b - a * c;
        if (delta < 0)
            return false;
        float sqrtdelta = std::sqrt(delta);

        // Find the nearest root that lies in the acceptable range.
        float root = (-b - sqrtdelta) / a;
        if (root < t_min || t_max < root) {
            root = (-b + sqrtdelta) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.hit_point = r.at(rec.t);
        vec3 outward_normal = (rec.hit_point - this->center) / this->radius;
        rec.set_face_normal(r, outward_normal);
        rec.material_ptr = this->material_ptr;

        return true;
    }

    bool sphere::scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const {
        return true;
    }

}
