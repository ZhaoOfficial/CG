#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_

#include "hittable.hpp"
#include "vec3.hpp"

namespace RayTracing {

    class Sphere : public Hittable {
    public:
        Sphere(Point3 center, float radius, std::shared_ptr<Material> mat_ptr);

        virtual bool hit(
            const Ray& r, float t_min, float t_max, HitRecord& rec
        ) const override;

        virtual bool bounding_box(
            float t_min, float t_max, AABB& aabb
        ) const override;

        virtual bool scatter(
            const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered
        ) const override;

        static void getSphereUV(float& u, float& v, const Point3& p);

    public:
        Point3 center;
        float radius;
        std::shared_ptr<Material> mat_ptr;
    };

}

#endif // !_SPHERE_HPP_