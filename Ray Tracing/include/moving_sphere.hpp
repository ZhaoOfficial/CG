#ifndef _MOVING_SPHERE_H_
#define _MOVING_SPHERE_H_

#include <memory>

#include "hittable.hpp"
#include "material.hpp"

namespace RayTracing {

    class MovingSphere : public Hittable {
    public:
        MovingSphere();
        MovingSphere(
            Point3 start_center, Point3 end_center,
            float start_time, float end_time,
            float radius, std::shared_ptr<Material> mat_ptr
        );

        Point3 current_center(float time) const;

        virtual bool hit(
            const Ray& r, float t_min, float t_max, HitRecord& rec
        ) const override;

        virtual bool bounding_box(
            float time0, float time1, AABB& aabb
        ) const override;

        virtual bool scatter(
            const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered
        ) const override;

    public:
        Point3 start_center;
        Point3 end_center;
        float start_time;
        float end_time;
        float radius;
        std::shared_ptr<Material> mat_ptr;
    };

}

#endif // !_MOVING_SPHERE_H
