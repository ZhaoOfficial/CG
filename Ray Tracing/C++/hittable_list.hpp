#ifndef _HITTABLE_LIST_HPP_
#define _HITTABLE_LIST_HPP_

#include "hittable.hpp"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    virtual bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const override;

public:
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (const auto& object : this->objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

bool hittable_list::scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const {
    return rec.material_ptr->scatter(r, rec.p, rec.normal, rec.front_face, attenuation, scattered);
}

#endif // !_HITTABLE_LIST_HPP_