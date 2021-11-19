#ifndef _HITTABLE_LIST_HPP_
#define _HITTABLE_LIST_HPP_

#include <memory>
#include <vector>

#include "hittable.hpp"

namespace RayTracing {

    class HittableList {
    public:
        HittableList();
        HittableList(std::shared_ptr<Hittable> object);

        void clear();
        void add(std::shared_ptr<Hittable> object);

        bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
        bool bounding_box(float time0, float time1, AABB& aabb) const;
        bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const;

    public:
        std::vector<std::shared_ptr<Hittable>> objects;
    };

}

#endif // !_HITTABLE_LIST_HPP_