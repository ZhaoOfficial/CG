#ifndef _HITTABLE_HPP_
#define _HITTABLE_HPP_

#include "axis_aligned_bounding_box.hpp"
#include "ray.hpp"
#include "material.hpp"
#include "utility.hpp"

namespace RayTracing {

    struct HitRecord {
        float t{ 0.0f };
        float u{ 0.0f };
        float v{ 0.0f };
        Point3 hit_point{ 0.0f, 0.0f, 0.0f };
        Vec3 normal{ 0.0f, 0.0f, 0.0f };
        std::shared_ptr<Material> mat_ptr{ nullptr };
        bool front_face;

        // outward_normal always point outside of the sphere surface
        inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
            // dot < 0.0f, the ray is arriving to the sphere
            // dot > 0.0f, the ray is leaving from the sphere
            front_face = (dot(r.direction, outward_normal) < 0.0f);
            normal = front_face ? outward_normal : -outward_normal;
        }
    };

    class Hittable {
    public:
        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const = 0;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
    };

    class Translate : public Hittable {
    public:
        Translate(std::shared_ptr<Hittable> hit_ptr, const Vec3& displacement);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;

    public:
        std::shared_ptr<Hittable> hit_ptr;
        Vec3 displacement;
    };

    class YRotate : public Hittable {
    public:
        YRotate(std::shared_ptr<Hittable> hit_ptr, float angle);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;

    public:
        std::shared_ptr<Hittable> hit_ptr;
        float sin_theta;
        float cos_theta;
        bool has_box;
        AABB aabb;
    };
}

#endif // !_HITTABLE_H_