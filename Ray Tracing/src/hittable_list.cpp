#include <memory>
#include <vector>

#include "hittable_list.hpp"

namespace RayTracing {

	HittableList::HittableList() {}
	HittableList::HittableList(std::shared_ptr<Hittable> object) { add(object); }

	void HittableList::clear() { this->objects.clear(); }
	void HittableList::add(std::shared_ptr<Hittable> object) { objects.push_back(object); }

    bool HittableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (auto& object : this->objects) {
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

    bool HittableList::bounding_box(float time0, float time1, AABB& aabb) const {
        if (this->objects.empty())
            return false;
        
        AABB temp_aabb;
        bool first_aabb = true;

        for (auto& object : this->objects) {
            if (object->bounding_box(time0, time1, temp_aabb) == false) {
                return false;
            }
            aabb = first_aabb ? temp_aabb : surrounding_box(aabb, temp_aabb);
            first_aabb = false;
        }

        return true;
    }


    bool HittableList::scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const {
        return rec.mat_ptr->scatter(r, rec.u, rec.v, rec.hit_point, rec.normal, rec.front_face, attenuation, scattered);
    }
}
