#ifndef _HITTABLE_LIST_HPP_
#define _HITTABLE_LIST_HPP_

#include <memory>
#include <vector>

#include "hittable.hpp"

namespace RayTracing {

    class hittable_list {
    public:
        hittable_list();
        hittable_list(std::shared_ptr<hittable> object);

        void clear();
        void add(std::shared_ptr<hittable> object);

        bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        bool scatter(const ray& r, hit_record& rec, color& attenuation, ray& scattered) const;

    public:
        std::vector<std::shared_ptr<hittable>> objects;
    };

}

#endif // !_HITTABLE_LIST_HPP_