#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_

#include "utility.hpp"
#include "hittable.hpp"
#include "vec3.hpp"

class sphere : public hittable {
public:
    sphere() {}
    sphere(point3 cen, float r, shared_ptr<material> m) : center(cen), radius(r), material_ptr(m) {};

    virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const override;

public:
    point3 center;
    float radius;
    shared_ptr<material> material_ptr;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - this->center;
    float a = r.direction().length_squared();
    float b = r.direction().dot(oc);
    float c = oc.length_squared() - this->radius * this->radius;

    float delta = b * b - a * c;
    if (delta < 0)
        return false;
    float sqrtdelta = sqrt(delta);

    // Find the nearest root that lies in the acceptable range.
    float root = (-b - sqrtdelta) / a;
    if (root < t_min || t_max < root) {
        root = (-b + sqrtdelta) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - this->center) / this->radius;
    rec.set_face_normal(r, outward_normal);
    rec.material_ptr = this->material_ptr;
    
    return true;
}

bool sphere::scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const {
    return true;
}


#endif // !_SPHERE_HPP_