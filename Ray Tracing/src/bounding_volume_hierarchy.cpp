#include <algorithm>
#include <functional>

#include "bounding_volume_hierarchy.hpp"
#include "utility.hpp"

namespace RayTracing {

    BoundingVolumeHierarchy::BoundingVolumeHierarchy() {}

    BoundingVolumeHierarchy::BoundingVolumeHierarchy(const HittableList& list, float time0, float time1)
    : BoundingVolumeHierarchy(list.objects, 0, list.objects.size(), time0, time1) {}

    BoundingVolumeHierarchy::BoundingVolumeHierarchy(const std::vector<std::shared_ptr<Hittable>>& objects, size_t start, size_t finish, float time0, float time1) {
        std::vector<std::shared_ptr<Hittable>> obj_copy = objects;

        size_t object_span = finish - start;
        int axis = uniform_int(0, 2);

        if (object_span == 1) {
            this->left = obj_copy[start];
            this->right = obj_copy[start];
        }
        else if (object_span == 2) {
            if (box_compare(objects[start], obj_copy[start + 1], axis)) {
                this->left = obj_copy[start];
                this->right = obj_copy[start + 1];
            }
            else {
                this->left = obj_copy[start + 1];
                this->right = obj_copy[start];
            }
        }
        else {
            std::sort(obj_copy.begin() + start, obj_copy.begin() + finish, std::bind(box_compare, std::placeholders::_1, std::placeholders::_2, axis));

            size_t middle = start + object_span / 2;
            this->left = std::make_shared<BVH>(obj_copy, start, middle, time0, time1);
            this->right = std::make_shared<BVH>(obj_copy, middle, finish, time0, time1);
        }

        AABB aabb_left, aabb_right;

        this->left->bounding_box(time0, time1, aabb_left);
        this->right->bounding_box(time0, time1, aabb_right);

        this->bvh_box = surrounding_box(aabb_left, aabb_right);

    }

    bool BoundingVolumeHierarchy::box_compare(
        const std::shared_ptr<Hittable>& a,
        const std::shared_ptr<Hittable>& b,
        int axis
    ) {
        AABB aabb0;
        AABB aabb1;

        a->bounding_box(0, 0, aabb0);
        b->bounding_box(0, 0, aabb1);

        return aabb0.minimum[axis] < aabb1.minimum[axis];
    }

    bool BoundingVolumeHierarchy::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (this->bvh_box.hit(r, t_min, t_max) == false)
            return false;
        
        bool hit_left = this->left->hit(r, t_min, t_max, rec);
        bool hit_right= this->right->hit(r, t_min, (hit_left ? rec.t : t_max), rec);

        return hit_left || hit_right;
    }

    bool BoundingVolumeHierarchy::bounding_box(float t_min, float t_max, AABB& aabb) const {
        aabb = this->bvh_box;
        return true;
    }

    bool BoundingVolumeHierarchy::scatter(
        const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered
    ) const {
        return false;
    }

}
