#include <memory>
#include <vector>

#include "hittable_list.hpp"

namespace RayTracing {

	hittable_list::hittable_list() {}
	hittable_list::hittable_list(std::shared_ptr<hittable> object) { add(object); }

	void hittable_list::clear() { this->objects.clear(); }
	void hittable_list::add(std::shared_ptr<hittable> object) { objects.push_back(object); }

    bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (const auto& object : this->objects) {
            // a ray may hit many objects
            // choose the closest one to hit
            if (object->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    bool hittable_list::scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const {
        return rec.material_ptr->scatter(r, rec.hit_point, rec.normal, rec.front_face, attenuation, scattered);
    }
}
