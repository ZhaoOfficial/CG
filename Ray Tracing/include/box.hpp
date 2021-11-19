#ifndef _BOX_HPP_
#define _BOX_HPP_

#include "axis_aligned_rectangle.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"

namespace RayTracing {

    class Box : public Hittable {
    public:
        Box();
        Box(const Point3& minimum, const Point3& maximum, std::shared_ptr<Material> mat_ptr);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;

    public:
        Point3 minimum;
        Point3 maximum;
        HittableList faces;
    };
}

#endif // !_BOX_HPP_
