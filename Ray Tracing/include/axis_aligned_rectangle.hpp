#ifndef _AXIS_ALIGNED_RECTANGLE_HPP_
#define _AXIS_ALIGNED_RECTANGLE_HPP_

#include "hittable.hpp"

namespace RayTracing {

    class XYRectangle : public Hittable {
    public:
        XYRectangle();
        XYRectangle(float x0, float x1, float y0, float y1, float z, std::shared_ptr<Material> mat_ptr);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;
    
    public:
        float x0, x1, y0, y1, z;
        std::shared_ptr<Material> mat_ptr;
    };

    class XZRectangle : public Hittable {
    public:
        XZRectangle();
        XZRectangle(float x0, float x1, float z0, float z1, float y, std::shared_ptr<Material> mat_ptr);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;
    
    public:
        float x0, x1, z0, z1, y;
        std::shared_ptr<Material> mat_ptr;
    };

    class YZRectangle : public Hittable {
    public:
        YZRectangle();
        YZRectangle(float y0, float y1, float z0, float z1, float x, std::shared_ptr<Material> mat_ptr);

        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
        virtual bool bounding_box(float time0, float time1, AABB& aabb) const override;
        virtual bool scatter(const Ray& r, HitRecord& rec, Color& attenuation, Ray& scattered) const override;
    
    public:
        float y0, y1, z0, z1, x;
        std::shared_ptr<Material> mat_ptr;
    };

}

#endif // !_AXIS_ALIGNED_RECTANGLE_HPP
